#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
Particle Transformer for Z -> tau tau Selection
=============================================================================

This script combines/modifies several sources:

1. @InProceedings{Qu:2022mxj,
    author = "Qu, Huilin and Li, Congqiao and Qian, Sitian",
    title = "{Particle Transformer} for Jet Tagging",
    booktitle = "{Proceedings of the 39th International Conference on Machine Learning}",
    pages = "18281--18292",
    year = "2022",
    eprint = "2202.03772",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph"
}

2. Original code by the author (Federico Melis).

3. Code under MIT License from the repository https://github.com/jet-universe/particle_transformer .

4. Code under GPLv3 License from the repository https://github.com/HEP-KBFI/ml-tau-model .

=============================================================================
"""

import os
import math
import copy
import warnings
import contextlib
from functools import partial

import numpy as np
import awkward as ak # Used for calculations on arrays where each event has a different number of particles
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ====================================================================
# START OF PART 1: PARTICLE TRANSFORMER
# ====================================================================
# Code source: https://github.com/HEP-KBFI/ml-tau-model 
# Original file: mltau/models/ParticleTransformer.py
#
# Modifications author: Federico Melis (2026).
# ====================================================================

#=====================================================================
# 1.1 PARTICLE TRANSFORMER FUNCTIONS
#=====================================================================

@torch.jit.script
def delta_phi(a, b): # Identical to mltau/models/ParticleTransformer.py
    """
    Compute the difference in azimuthal angle, wrapped to [-pi, pi].

    Args:
        a (torch.Tensor): First angle tensor.
        b (torch.Tensor): Second angle tensor.

    Returns:
        torch.Tensor: Delta phi tensor.
    """
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script 
def delta_r2(eta1, phi1, eta2, phi2):  # Identical to mltau/models/ParticleTransformer.py
    """
    Compute the squared angular distance in eta-phi space.

    Args:
        eta1 (torch.Tensor): Eta of first point.
        phi1 (torch.Tensor): Phi of first point.
        eta2 (torch.Tensor): Eta of second point.
        phi2 (torch.Tensor): Phi of second point.

    Returns:
        torch.Tensor: Delta R squared tensor.
    """
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


def to_pt2(x, eps=1e-8):  # Identical to mltau/models/ParticleTransformer.py
    """
    Compute transverse momentum squared from 4-momentum.

    Args:
        x (torch.Tensor): 4-momentum tensor of shape (N, 4, ...).
        eps (float, optional): Minimum value to clamp. Defaults to 1e-8.

    Returns:
        torch.Tensor: Transverse momentum squared.
    """
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):  # Identical to mltau/models/ParticleTransformer.py
    """ 
    Compute mass squared from 4-momentum.

    Args:
        x (torch.Tensor): 4-momentum tensor of shape (N, 4, ...).
        eps (float, optional): Minimum value to clamp. Defaults to 1e-8.

    Returns:
        torch.Tensor: Mass squared.
    """
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2

def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False): # Modified from mltau/models/ParticleTransformer.py
    """
        Convert 4-momentum to pt, rapidity, phi, and optionally mass.
    
        Args:
            x (torch.Tensor): 4-momentum tensor of shape (N, 4, ...).
            return_mass (bool, optional): Whether to include mass. Defaults to True.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            for_onnx (bool, optional): Use ONNX-compatible functions. Defaults to False.
    
        Returns:
            torch.Tensor: Converted features.
        """
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # MODIFICATION: 
    # Clamped the numerator as well: if a photon (m=0) travels almost parallel to the beam in the negative hemisphere,
    # E + pz would be close to 0, avoiding values <= 0.
    rapidity = 0.5 * torch.log((energy + pz).clamp(min=1e-20) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)

def pairwise_lv_fts(xi, xj, to_ptXXXphim, num_outputs=4, eps=1e-8, for_onnx=False): # Modified from mltau/models/ParticleTransformer.py 
    """
        Compute pairwise Lorentz vector features between particles.
    
        Args:
            xi (torch.Tensor): First particle 4-momenta.
            xj (torch.Tensor): Second particle 4-momenta.
            to_ptXXXphim (callable): Function to convert 4-momentum.
            num_outputs (int, optional): Number of output features. Defaults to 4.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            for_onnx (bool, optional): Use ONNX-compatible operations. Defaults to False.
    
        Returns:
            torch.Tensor: Pairwise features.
        """

    # MODIFICATION: 
    # Removed initial if statements since only num_outputs=4 is relevant here.
    pti, rap_or_thetai, phii = to_ptXXXphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rap_or_thetaj, phij = to_ptXXXphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    
    delta = delta_r2(rap_or_thetai, phii, rap_or_thetaj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    
    ptmin = torch.minimum(pti, ptj)
    lnkt = torch.log((ptmin * delta).clamp(min=eps))
    lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
    
    xij = xi + xj
    lnm2 = torch.log(to_m2(xij, eps=eps))
    
    return torch.cat([lnkt, lnz, lndelta, lnm2], dim=1)

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0): # Modified from mltau/models/ParticleTransformer.py
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        # MODIFICATION:
        # Removed the if statement

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lo = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

         # Uniformly fill tensor with values from [l, u], then translate to
         # [2lo-1, 2up-1].
        tensor.uniform_(2 * lo - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

         # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

#=====================================================================
# 1.2 CLASSI DEL PARTICLE TRANSFORMER
#=====================================================================

# Blocco "PARTICLES->EMBEDDING" di fig. 3a del paper: codifica delle singole particelle
# MLP a 3 strati che trasforma le 5 variabili di ogni particella (px, py, pz, E, q) in un vettore di 128 elementi

class Embed(nn.Module): # Identical to mltau/models/ParticleTransformer.py
    """
        Embedding layer with optional batch normalization and MLP.
    
        Args:
            input_dim (int): Input dimension.
            dims (list): List of dimensions for the MLP layers.
            normalize_input (bool, optional): Whether to normalize input. Defaults to True.
            activation (str, optional): Activation function. Defaults to "gelu".
        """
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu"):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend(
                [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
            input_dim = dim
        self.embed = nn.Sequential(*module_list)


    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)

# "INTERACTIONS->EMBEDDING" block: pairwise particle interaction encoding
# Builds the interaction matrix U for each particle pair

class PairEmbed(nn.Module): # Modified from mltau/models/ParticleTransformer.py
    """
    Pairwise embedding layer for particle interactions.

    Args:
        to_ptXXXphim (callable): Function to convert 4-momentum.
        pairwise_lv_dim (int): Dimension of pairwise Lorentz features.
        pairwise_input_dim (int): Input dimension for pairwise features.
        dims (list): Dimensions for embedding layers.
        remove_self_pair (bool, optional): Whether to remove self-pairs. Defaults to False.
        use_pre_activation_pair (bool, optional): Use pre-activation. Defaults to True.
        mode (str, optional): Mode 'sum' or 'concat'. Defaults to "sum".
        normalize_input (bool, optional): Normalize input. Defaults to True.
        activation (str, optional): Activation function. Defaults to "gelu".
        eps (float, optional): Epsilon. Defaults to 1e-8.
    """
    # MODIFICATION:
    # Removed unnecessary onnx compatibility block
    # Removed unused 'if self.mode == "concat"' blocks

    def __init__(
        self,
        to_ptXXXphim,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        mode="sum",
        normalize_input=True,
        activation="gelu",
        eps=1e-8,
    ):
        super().__init__()

        self.to_ptXXXphim = to_ptXXXphim
        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.pairwise_lv_fts = partial(pairwise_lv_fts, to_ptXXXphim=self.to_ptXXXphim, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=False)
        self.out_dim = dims[-1]

        if self.mode == "sum" and pairwise_lv_dim > 0:
            input_dim = pairwise_lv_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        with torch.no_grad():
            batch_size, _, seq_len = x.size()
            if self.is_symmetric:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0, device=x.device)
                x_exp = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                xi = x_exp[:, :, i, j]
                xj = x_exp[:, :, j, i]
                x_pair = self.pairwise_lv_fts(xi, xj)
            
        elements = self.embed(x_pair)
        
        if self.is_symmetric:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y

# "PARTICLE ATTENTION BLOCK" 

class Block(nn.Module): # Modified from mltau/models/ParticleTransformer.py
    """
    Transformer block with multi-head attention and feed-forward network.

    Args:
        embed_dim (int, optional): Embedding dimension. Defaults to 128.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        ffn_ratio (int, optional): Feed-forward ratio. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        attn_dropout (float, optional): Attention dropout. Defaults to 0.1.
        activation_dropout (float, optional): Activation dropout. Defaults to 0.1.
        add_bias_kv (bool, optional): Add bias to kv. Defaults to False.
        activation (str, optional): Activation function. Defaults to "gelu".
        scale_fc (bool, optional): Scale FC. Defaults to True.
        scale_attn (bool, optional): Scale attention. Defaults to True.
        scale_heads (bool, optional): Scale heads. Defaults to True.
        scale_resids (bool, optional): Scale residuals. Defaults to True.
    """
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        add_bias_kv=False,
        activation="gelu",
        scale_fc=True,
        scale_attn=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv)
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True)
            if scale_heads
            else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True)
            if scale_resids
            else None
        )

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
                Args:
                    x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
                    x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
                    padding_mask (ByteTensor, optional): binary
                        ByteTensor of shape `(batch, seq_len)` where padding
                        elements are indicated by ``1``.
        
                Returns:
                    encoded output of shape `(seq_len, batch, embed_dim)`
                """   
        # MODIFICATION:
        # Added a_mask = attn_mask.to(dtype=x.dtype), self.c_attn.to(dtype=x.dtype), and self.w_resid.to(dtype=residual.dtype).
        # When using mixed precision (AMP), x becomes float16, while attn_mask and c_attn/w_resid parameters remain float32. 
        # Without this conversion, PyTorch may throw incompatible type errors inside MultiheadAttention.

        if x_cls is not None:
            with torch.no_grad():
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]
        else:
            residual = x
            x = self.pre_attn_norm(x)
            kp_mask = padding_mask.to(dtype=x.dtype).masked_fill_(padding_mask, float("-inf"))
            a_mask = attn_mask.to(dtype=x.dtype) if attn_mask is not None else None
            x = self.attn(x, x, x, key_padding_mask=kp_mask, attn_mask=a_mask)[0]
            
        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn.to(dtype=x.dtype))
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid.to(dtype=residual.dtype), residual)
        x += residual
        return x

# PARTICLE TRANSFORMER

class ParticleTransformer(nn.Module): # Modified from mltau/models/ParticleTransformer.py
    # MODIFICATION:
    # Removed redundant inputs and checks. 
    # The final layer is written directly as self.fc = nn.Linear(embed_dim, num_classes) to output the 4 scores 
    # for the 4 classes (muons, electrons, taus, hadrons).
    """
        Particle Transformer model for jet tagging.
    
        Args:
            input_dim (int): Input feature dimension.
            num_classes (int, optional): Number of output classes.
            pair_input_dim (int, optional): Pairwise input dimension. Defaults to 4.
            use_pre_activation_pair (bool, optional): Use pre-activation. Defaults to True.
            embed_dims (list, optional): Embedding dimensions. Defaults to [256, 512, 256].
            pair_embed_dims (list, optional): Pair embedding dims. Defaults to [64, 64, 64].
            num_heads (int, optional): Number of heads. Defaults to 8.
            num_layers (int, optional): Number of layers. Defaults to 8.
            num_cls_layers (int, optional): Number of class layers. Defaults to 2.
            use_amp (bool, optional): Use AMP. Defaults to False.
        
        """
    def __init__(self, 
                 input_dim, 
                 num_classes, 
                 pair_input_dim=4, 
                 use_pre_activation_pair=False, 
                 embed_dims=[128, 512, 128], 
                 pair_embed_dims=[64, 64, 64], 
                 num_heads=8, 
                 num_layers=8, 
                 num_cls_layers=2, 
                 use_amp=False):
    
        super().__init__()
        self.use_amp = use_amp
        embed_dim = embed_dims[-1]
        
        cfg_block = dict(embed_dim=embed_dim, 
                         num_heads=num_heads, 
                         ffn_ratio=4, 
                         dropout=0.1, 
                         attn_dropout=0.1, 
                         activation_dropout=0.1, 
                         add_bias_kv=False, 
                         activation="gelu", 
                         scale_fc=True, 
                         scale_attn=True, 
                         scale_heads=True, 
                         scale_resids=True)
        
        cfg_cls_block = dict(embed_dim=embed_dim, 
                             num_heads=num_heads, 
                             ffn_ratio=4, 
                             dropout=0, 
                             attn_dropout=0, 
                             activation_dropout=0, 
                             add_bias_kv=False, 
                             activation="gelu", 
                             scale_fc=True, 
                             scale_attn=True, 
                             scale_heads=True, 
                             scale_resids=True)

        self.embed = Embed(input_dim, embed_dims, activation="gelu")
        self.pair_embed = PairEmbed(to_ptrapphim, 
                                    pair_input_dim, 
                                    0, 
                                    pair_embed_dims + [cfg_block["num_heads"]], 
                                    remove_self_pair=False, 
                                    use_pre_activation_pair=use_pre_activation_pair)
        
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.fc = nn.Linear(embed_dim, num_classes)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, cand_features, cand_kinematics_pxpypze, cand_mask):
        cand_mask = cand_mask.type(torch.bool)
        padding_mask = ~cand_mask.squeeze(1)
        amp_ctx = torch.amp.autocast("cuda") if self.use_amp else contextlib.nullcontext()
        
        with amp_ctx:
            num_particles = cand_features.size(-1)
            cand_features_embed = self.embed(cand_features).masked_fill(~cand_mask.permute(2, 0, 1), 0)
            
            attn_mask = self.pair_embed(cand_kinematics_pxpypze).view(-1, num_particles, num_particles)

            for block in self.blocks:
                cand_features_embed = block(cand_features_embed, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            cls_tokens = self.cls_token.expand(1, cand_features_embed.size(1), -1)
            for block in self.cls_blocks:
                cls_tokens = block(cand_features_embed, x_cls=cls_tokens, padding_mask=padding_mask)
            
            x_cls = self.norm(cls_tokens).squeeze(0)
            output = self.fc(x_cls)
            return output

# ====================================================================
# END OF PART 1
# ====================================================================


# ====================================================================
# START OF PART 2: DATA PREPROCESSING, PADDING, AND TENSOR CREATION
# ====================================================================
# Source: Original code by the author (Federico Melis).
# ====================================================================

# Fix seed for reproducibility: identical weight initialization and data splitting at every run
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Angular cut
COS_THETA_CUT = 0.7  # cos(120millirad)=0.993. Also tested with COS_THETA_CUT = 0.7
MAX_PARTICLES = 50

# Network hyperparameters

EPOCHS = 30
BATCH_SIZE = 512 # Number of events processed simultaneously before updating weights
LEARNING_RATE = 0.001

INPUT_DIR = "output_root_files_ParT" # Folder containing the root files output from the analysis script
OUTPUT_DIR = "neuralnet_ParT_output_files_cos07" # NB: must be changed if using cos07
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCESSES = {
    "p8_ee_Ztautau_ecm91.root": 0, # Class 0: Tau signal
    "p8_ee_Zmumu_ecm91.root"  : 1, # Class 1: Muon background
    "p8_ee_Zee_ecm91.root"    : 2, # Class 2: Electron background
    "p8_ee_Zud_ecm91.root"    : 3, # Class 3: Hadronic background (u, d)
    "p8_ee_Zss_ecm91.root"    : 3, # Class 3: Hadronic background (s)
    "p8_ee_Zcc_ecm91.root"    : 3, # Class 3: Hadronic background (c)
    "p8_ee_Zbb_ecm91.root"    : 3  # Class 3: Hadronic background (b)
}
CLASS_NAMES = ["Tau (Sig)", "Muon (Bkg)", "Electron (Bkg)", "Hadron (Bkg)"] 

BRANCHES_TO_READ = ["RP_px", "RP_py", "RP_pz", "RP_e", "RP_charge", "cos_theta_thrust"]

print(">>> [1/5] Extracting Particle Clouds and Padding...")

data_by_class = {0: [], 1: [], 2: [], 3: []}
total_initial_by_class = {0: 0, 1: 0, 2: 0, 3: 0}
total_selected_by_class = {0: 0, 1: 0, 2: 0, 3: 0}

def process_and_pad(file_path):
    with uproot.open(file_path) as file:
        tree = file["events"]
        # Read the 6 columns from the root file as Awkward Arrays since the RP_ branches are variable-length vectors event by event
        df = tree.arrays(BRANCHES_TO_READ, library="ak")
        
        # Count events before discarding those with insufficient information.
        # Count events inside the angular cut before applying the has_particles filter (i.e., before dropping empty events)
        mask_angle = df["cos_theta_thrust"] < COS_THETA_CUT
        n_initial = int(ak.sum(mask_angle))
        
        # Discard anomalous/empty events (as in MLP).
        # Count how many particles have E > 0 per event and ensure there is at least one.
        has_particles = ak.sum(df["RP_e"] > 0, axis=1) > 0
        mask_fiducial = mask_angle & has_particles # Apply both filters
        df = df[mask_fiducial]
        
        # Sort particles in each event from most to least energetic before truncating to MAX_PARTICLES. 
        # Ensures the 50 most relevant particles are kept for hadronic events with > 50 particles.
        sort_idx = ak.argsort(df["RP_e"], axis=1, ascending=False)

        # Pad events with fewer than 50 particles with 'None' up to 50. Truncate events with more than 50 particles.
        def pad_array(arr):
            sorted_arr = arr[sort_idx]
            padded = ak.pad_none(sorted_arr, MAX_PARTICLES, clip=True)
            # Replace None values with 0.0, convert to a NumPy array, and cast 0.0s to float32 
            # to reduce computational load.
            return ak.fill_none(padded, 0.0).to_numpy().astype(np.float32)
            
        px = pad_array(df["RP_px"])
        py = pad_array(df["RP_py"])
        pz = pad_array(df["RP_pz"])
        e  = pad_array(df["RP_e"])
        q  = pad_array(df["RP_charge"])
        cos_th = ak.to_numpy(df["cos_theta_thrust"]).astype(np.float32)
        return px, py, pz, e, q, cos_th, n_initial # Also return the number of pre-filter events

for file_name, label in PROCESSES.items():
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    px, py, pz, e, q, cos_th, n_init = process_and_pad(file_path)
    # Group the (px, py, pz, e, q, cos_th) tuples in data_by_class based on the class (0, 1, 2, 3)
    data_by_class[label].append((px, py, pz, e, q, cos_th))
    total_initial_by_class[label] += n_init # Initial number of events
    total_selected_by_class[label] += len(px) # Number of events surviving the filter
    print(f"  - {file_name}: {len(px)}/{n_init} events selected ({len(px)/n_init*100:.2f}%).")

# Calculate the fraction of events surviving the cut for each of the 4 classes (preselection efficiency)
presel_eff = np.array([
    total_selected_by_class[c] / total_initial_by_class[c] for c in range(4)
], dtype=np.float32)

print("\n>>>Class balancing...")
for label in data_by_class:
    if len(data_by_class[label]) > 0:
        data_by_class[label] = [np.concatenate([x[i] for x in data_by_class[label]]) for i in range(6)] # Merge hadronic events
        print(f"  - Total events ready for Class {label}: {len(data_by_class[label][0])}")

# Find the minimum number of events across the 4 classes (event count of the least populated class)
min_events = min(len(data_by_class[label][0]) for label in data_by_class if len(data_by_class[label]) > 0)
print(f"  -> Cutting all classes to exactly {min_events} events for 100% balance.")

# Extract 'min_events' from each class and create the final dataset.
# Concatenate all balanced samples and shuffle the entire dataset.

X_px, X_py, X_pz, X_e, X_q, X_costh, y_all = [], [], [], [], [], [], []
for label, arrays in data_by_class.items():
    if len(arrays) == 0:
        continue
    idx = np.random.choice(len(arrays[0]), min_events, replace=False)
    X_px.append(arrays[0][idx])
    X_py.append(arrays[1][idx])
    X_pz.append(arrays[2][idx])
    X_e.append(arrays[3][idx])
    X_q.append(arrays[4][idx])
    X_costh.append(arrays[5][idx])
    y_all.append(np.full(min_events, label))

y_numpy = np.concatenate(y_all)
cos_theta_numpy = np.concatenate(X_costh) # 1D array containing 1 number per event (shape: total_events,)

# Expand cos_theta_thrust across all true particles in the event (shape: total_events, 50)
e_concat = np.concatenate(X_e)
# Use newaxis to transform cos_theta_numpy into a column (shape: total_events, 1) ready to be broadcasted to all 50 particles of the event.
# Where there is padding (E == 0), keep 0.0 to remain consistent with the other 5 variables. If the particle is real (E > 0), assign the 
# event's cos_theta_thrust value.
costh_2d = np.where(e_concat > 0, cos_theta_numpy[:, np.newaxis], 0.0)

# Stack (px, py, pz, e, q, cos_theta) and create the 3D tensor of shape (total_events, 6, 50)
cand_features = np.stack([
    np.concatenate(X_px), # channel 0
    np.concatenate(X_py), # channel 1
    np.concatenate(X_pz), # channel 2
    e_concat, # channel 3
    np.concatenate(X_q), # channel 4
    costh_2d # channel 5
], axis=1).astype(np.float32)

# Look at channel 3 (RP_e energy): if E > 0 there is a real particle, where it is 0.0 it's padding. 
# The result has shape (total_events, 1, 50)
cand_mask = np.expand_dims((cand_features[:, 3, :] > 0).astype(np.float32), axis=1)

# Free RAM occupied by intermediate lists
del data_by_class, X_px, X_py, X_pz, X_e, X_q, X_costh, y_all

print("\n>>> [2/5] Creating PyTorch Datasets...")

# Split into Train (70%), Temporary (30%)
indices = np.arange(len(y_numpy))
idx_train, idx_temp, y_train, y_temp = train_test_split(
    indices, y_numpy, test_size=0.30, random_state=RANDOM_SEED, stratify=y_numpy # stratify=y_numpy ensures the 25% proportion per class is maintained in each set
)
# Split Temporary into Validation (15%) and Test (15%)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

def to_tensor(idx):
    return (torch.from_numpy(cand_features[idx]),
            torch.from_numpy(cand_mask[idx]))

# Data flow management during training.
# shuffle=True only for the train set: randomly shuffles event order at the start of each epoch (not needed for test and val).
train_loader = DataLoader(TensorDataset(*to_tensor(idx_train), torch.tensor(y_train, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(*to_tensor(idx_val), torch.tensor(y_val, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(*to_tensor(idx_test), torch.tensor(y_test, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)

print(f"  - Training Set:   {len(idx_train)} events")
print(f"  - Validation Set: {len(idx_val)} events")
print(f"  - Test Set:       {len(idx_test)} events")

# ====================================================================
# END OF PART 2
# ====================================================================


# =====================================================================
# START OF PART 3: TRAINING SETUP, TRAINING LOOP, AND HYPERPARAMETERS
# =====================================================================
# Sources:
#
# Architecture hyperparameters: 
# Source: https://github.com/jet-universe/particle_transformer 
# Original file: networks/example_ParticleTransformer.py
#
# Optimizer (AdamW) and Scheduler (OneCycleLR): 
# Source: https://github.com/HEP-KBFI/ml-tau-model 
# Original file: mltau/models/SingleParTau_module.py6).
#
# Training Loop Structure:
# Source: Original code by the author (Federico Melis).
# ====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modifications to networks/example_ParticleTransformer.py:
# Extracted only the necessary parameters from the 'cfg' dictionary (pair_input_dim=4, use_pre_activation_pair=False, embed_dims=[128, 512, 128],
# pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8, num_cls_layers=2) and passed directly to 'ParticleTransformer',
# removing the 'ParticleTransformerWrapper' class. Modified the number of inputs from 5 to 6 to include cos(theta_thrust).
model = ParticleTransformer(
    input_dim=6, # px, py, pz, e, q, cos(theta thrust)
    num_classes=4, 
    pair_input_dim=4, 
    use_pre_activation_pair=False,     
    embed_dims=[128, 512, 128], 
    pair_embed_dims=[64, 64, 64],      
    num_heads=8, 
    num_layers=8, 
    num_cls_layers=2,                  
    use_amp=False  # Autocast is handled only once in the outer loop during training/validation
).to(device)

# Extracted the configuration for 'AdamW' (weight_decay=1e-2) and 'OneCycleLR' (anneal_strategy="cos") 
# from the PyTorch Lightning 'configure_optimizers' method in mltau/models/SingleParTau_module.py to use in a native PyTorch loop.
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
total_steps = len(train_loader) * EPOCHS
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total_steps, anneal_strategy="cos")
# Enable the gradient scaler with mixed precision (AMP) on GPU
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
step_count = 0

print("\n>>> [3/5] Training...")
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')
best_saved_epoch = 0
best_model_path = os.path.join(OUTPUT_DIR, "part_tau_model_best.pth")

for epoch in range(EPOCHS):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for feat, mask, labels in train_loader:
        feat, mask, labels = feat.to(device), mask.to(device), labels.to(device)
        # Take the first 4 variables (px, py, pz, E) from feat without allocating new memory.
        # 'kine' goes to PairEmbed while the full 'feat' goes to Embed.
        kine = feat[:, :4, :]
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(feat, kine, mask)
                loss = criterion(outputs, labels)
            # Multiply the loss by a scale factor before computing derivatives, preventing small gradients from vanishing
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # When using GradScaler, if infinite gradients appear, scaler.step(optimizer) skips the weight update. I want
            # the scheduler to step only if GradScaler did not skip optimizer.step(), i.e., when the weights have been successfully updated.
            if scaler.get_scale() >= old_scale and step_count < total_steps:
                scheduler.step()
                step_count += 1
        else:
            outputs = model(feat, kine, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step_count < total_steps:
                scheduler.step()
                step_count += 1

        run_loss += loss.item() * feat.size(0) # Sum mean error by 512 (batch size) to get the total epoch error 
        _, predicted = torch.max(outputs.data, 1) # Returns the maximum logit value and the corresponding index (0,1,2,3)
        total += labels.size(0)  # Keep track of the total events seen so far
        correct += (predicted == labels).sum().item() # Count how many events out of the 512 were correctly predicted
        
    t_loss = run_loss / total
    t_acc = correct / total

    # Validation phase
    model.eval()
    v_loss, v_corr, v_tot = 0.0, 0, 0
    with torch.no_grad():
        for feat, mask, labels in val_loader:
            feat, mask, labels = feat.to(device), mask.to(device), labels.to(device)
            kine = feat[:, :4, :]
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(feat, kine, mask)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(feat, kine, mask)
                loss = criterion(outputs, labels)
            v_loss += loss.item() * feat.size(0)
            _, predicted = torch.max(outputs.data, 1)
            v_tot += labels.size(0)
            v_corr += (predicted == labels).sum().item()
            
    v_loss, v_acc = v_loss / v_tot, v_corr / v_tot
    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(model.state_dict(), best_model_path)
        best_saved_epoch = epoch + 1
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch [{epoch+1}/{EPOCHS}] | Train L: {t_loss:.4f} Acc: {t_acc:.4f} | Val L: {v_loss:.4f} Acc: {v_acc:.4f}")

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "part_tau_model_last.pth"))
print(f"  -> Best model (Epoch {best_saved_epoch}) saved with Val Loss: {best_val_loss:.4f}")

# ====================================================================
# END OF PART 3
# ====================================================================


# ====================================================================
# START OF PART 4: EVALUATION, METRICS, AND PLOTS
# ====================================================================
# Source: Original code by the author (Federico Melis).
# ====================================================================

#=====================================================================
# 4.1 TEST SET EVALUATION
#=====================================================================
print("\n>>> [4/5] Testing and Generating Plots...")

# Using the best weights found
model.load_state_dict(torch.load(best_model_path))

model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for feat, mask, labels in test_loader:
        feat, mask, labels = feat.to(device), mask.to(device), labels.to(device)
        kine = feat[:, :4, :]
        outputs = model(feat, kine, mask)

        # Softmax to get probabilities between 0 and 1
        probs = torch.softmax(outputs, dim=1) # Convert the 4 output logits into a normalized probability distribution between 0 and 1
        _, preds = torch.max(outputs.data, 1) # Select the index of the class with the highest probability (final prediction)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Save the 2D array of all probabilities assigned to all test events
        # The first column contains the probability associated with the tau signal (needed for efficiency and purity plots)
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds) # Convert to numpy arrays
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Extraction of cos(theta) and visible energy for the test set events
test_cos_theta = cos_theta_numpy[idx_test]
# Visible energy is the sum of the energies (channel 3) of the event's particles
test_e_vis = np.sum(cand_features[idx_test, 3, :], axis=1)

# Definition of physical weights relative to cross-sections (Tau = 1.0)

# Weights calculated from official cross-sections FCCee_procDict_winter2023_IDEA.json
sigma_tau = 1476.580
CLASS_WEIGHTS = {
    0: 1.0,                                                 # Z -> tau+ tau- 
    1: 1462.080 / sigma_tau,                                # Z -> mu+ mu- 
    2: 1462.090 / sigma_tau,                                # Z -> e+ e- 
    3: (11870.5 + 5215.46 + 5215.46 + 6645.46) / sigma_tau  # Hadrons 
}

# Array of weights associated with each event in the test set
sample_weights = np.array([CLASS_WEIGHTS[label] for label in all_labels], dtype=np.float32)

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

#=====================================================================
# 4.2 PLOTS
#=====================================================================

# Learning curves (epoch-by-epoch trend of Loss (left) and Accuracy (right))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train Loss', color='blue', lw=2)
ax1.plot(history['val_loss'], label='Validation Loss', color='orange', lw=2)
ax1.set_title('ParT: Loss Function Evolution')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cross Entropy Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(history['train_acc'], label='Train Accuracy', color='blue', lw=2)
ax2.plot(history['val_acc'], label='Validation Accuracy', color='orange', lw=2)
ax2.set_title('ParT: Overall Accuracy Evolution')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "part_learning_curves.png"), dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrix normalized by TRUE LABEL (Efficiency) weighted by cross-section
cm_true = confusion_matrix(all_labels, all_preds, sample_weight=sample_weights, normalize='true')
disp = ConfusionMatrixDisplay(cm_true, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.3f')
plt.title('ParT: Confusion Matrix Norm. by True Label (Efficiency) - Cross-Section Weighted)')
plt.savefig(os.path.join(OUTPUT_DIR, "part_cm_weighted.png"), dpi=300, bbox_inches='tight')
plt.close()

# Efficiency confusion matrix (including discarded preprocessing events in the denominator)
# Multiply each row of the efficiency matrix by the "preselection efficiency" defined above
cm_true_abs = cm_true * presel_eff[:, np.newaxis]
disp_abs = ConfusionMatrixDisplay(cm_true_abs, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp_abs.plot(cmap=plt.cm.Blues, ax=ax, values_format='.3f')
plt.title('ParT: Confusion Matrix Norm. by True Label (Efficiency) - incl. Discarded Events)')
plt.savefig(os.path.join(OUTPUT_DIR, "part_cm_incl_discarded.png"), dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrix normalized by PREDICTED LABEL (Purity) weighted by cross-section
cm_pred = confusion_matrix(all_labels, all_preds, sample_weight=sample_weights, normalize='pred')
disp_pred = ConfusionMatrixDisplay(cm_pred, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp_pred.plot(cmap=plt.cm.Oranges, ax=ax, values_format='.3f')
plt.title('ParT: Confusion Matrix Norm. by Pred. Label (Purity) - Cross-Section Weighted)')
plt.savefig(os.path.join(OUTPUT_DIR, "part_cm_pred_weighted.png"), dpi=300, bbox_inches='tight')
plt.close()

# Histogram of the probability of being a tau
tau_scores = all_probs[:, 0]
fig, ax = plt.subplots(figsize=(9, 6))
colors = ['red', 'blue', 'green', 'orange']
for i in range(4):
    mask = (all_labels == i)
    ax.hist(tau_scores[mask], bins=50, histtype='step', lw=2, 
            label=f'True {CLASS_NAMES[i]}', color=colors[i], density=True)
ax.set_title('ParT Response: Tau Output Probability')
ax.set_xlabel('P(Tau)')
ax.set_ylabel('Normalized Density (A.U.)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "part_nn_score_response.png"), dpi=300, bbox_inches='tight')
plt.close()

# Efficiency and purity vs threshold
thresholds = np.linspace(0.01, 0.99, 100)  # thresholds is an array of 100 values from 0.01 to 0.99
effs, purs = [], []
total_tau = np.sum(all_labels == 0)

total_tau_weighted = np.sum(sample_weights[all_labels == 0])

for t in thresholds:
    sel = (tau_scores >= t)
    TP = np.sum(sample_weights[sel & (all_labels == 0)]) # True and false positives are weighted sums based on cross-section
    FP = np.sum(sample_weights[sel & (all_labels != 0)])
    effs.append(TP / total_tau_weighted if total_tau_weighted > 0 else 0)
    purs.append(TP / (TP + FP) if (TP + FP) > 0 else 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Efficiency and purity as a function of the threshold (left)
ax1.plot(thresholds, effs, label='Efficiency', color='green', lw=2)
ax1.plot(thresholds, purs, label='Purity', color='purple', lw=2)
ax1.set_title('ParT: Performance vs Decision Threshold')
ax1.set_xlabel('Decision Threshold on P(Tau)')
ax1.set_ylabel('Efficiency / Purity')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# ROC curve (purity vs efficiency) (right)
ax2.plot(effs, purs, color='darkred', lw=2)
ax2.set_title('ParT: Purity vs Efficiency Curve')
ax2.set_xlabel('Efficiency')
ax2.set_ylabel('Purity')
ax2.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "part_efficiency_purity_roc.png"), dpi=300, bbox_inches='tight')
plt.close()

# Efficiency and purity vs cos(theta) and visible energy

def plot_eff_pur_vs_var(var_array, bins, var_name, xlabel, filename, use_weights=True):
    """
    Calculates and plots the Efficiency and Purity of the Tau selection (class 0)
    as a function of cos_theta and E_vis
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    eff_list, eff_err_list = [], []
    pur_list, pur_err_list = [], []
    
    w = sample_weights if use_weights else np.ones_like(all_labels, dtype=np.float32)

    for i in range(len(bins) - 1):
        # Consider events whose visible energy or cosine fall in the current bin
        in_bin = (var_array >= bins[i]) & (var_array < bins[i+1])
        
        is_tau = (all_labels == 0) & in_bin
        pred_tau = (all_preds == 0) & in_bin
        
        tp_mask = is_tau & pred_tau
        
        # Unweighted counts
        n_true_tau = np.sum(is_tau)
        n_pred_tau = np.sum(pred_tau)
        
        # Weighted sums for efficiency and purity values
        W_true_tau = np.sum(w[is_tau])
        W_pred_tau = np.sum(w[pred_tau])
        W_tp = np.sum(w[tp_mask])
        
        # Efficiency calculation in the bin
        if W_true_tau > 0 and n_true_tau > 0:
            eff = W_tp / W_true_tau
            eff_err = np.sqrt(max(eff * (1.0 - eff), 0.0) / n_true_tau)
        else:
            eff, eff_err = np.nan, 0.0
            
        # Purity calculation in the bin (cross-section weighted)
        if W_pred_tau > 0 and n_pred_tau > 0:
            pur = W_tp / W_pred_tau
            # Purity error (binomial)
            pur_err = np.sqrt(max(pur * (1.0 - pur), 0.0) / n_pred_tau)
        else:
            pur, pur_err = np.nan, 0.0
            
        eff_list.append(eff)
        eff_err_list.append(eff_err)
        pur_list.append(pur)
        pur_err_list.append(pur_err)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(bin_centers, eff_list, yerr=eff_err_list, fmt='o-', color='green', 
                lw=2, capsize=3, label='Efficiency (Tau)')
    ax.errorbar(bin_centers, pur_list, yerr=pur_err_list, fmt='s-', color='purple', 
                lw=2, capsize=3, label='Purity (Tau, Cross Section weighted)')
    
    ax.set_title(f'ParT: Tau Efficiency & Purity vs {var_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Efficiency / Purity')
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Plot as a function of cos(theta_thrust) (50 bins between 0 and COS_THETA_CUT)
cos_bins = np.linspace(0.0, COS_THETA_CUT, 51)
plot_eff_pur_vs_var(
    var_array=test_cos_theta,
    bins=cos_bins,
    var_name=r'$\cos(\theta_{\mathrm{thrust}})$',
    xlabel=r'$\cos(\theta_{\mathrm{thrust}})$',
    filename="part_eff_pur_vs_costheta.png",
    use_weights=True
)

# Plot as a function of Visible Energy (20 bins between 0 and 100 GeV at ECM=91 GeV)
evis_bins = np.linspace(0.0, 100.0, 21)
plot_eff_pur_vs_var(
    var_array=test_e_vis,
    bins=evis_bins,
    var_name='Visible Energy',
    xlabel=r'$E_{\mathrm{vis}}$ [GeV]',
    filename="part_eff_pur_vs_evis.png",
    use_weights=True
)

# Purity for individual channels and tau signal contamination

def plot_all_channels_purity_vs_var(var_array, bins, var_name, xlabel, filename, use_weights=True):
    """
    Calculates and plots the Purity of all 4 channels (Tau, Muon, Electron, Hadron)
    as a function of a kinematic variable to see where each channel loses sensitivity.
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) # Used to position the points on the X axis at the midpoint of each bin
    w = sample_weights if use_weights else np.ones_like(all_labels, dtype=np.float32) # Weights array
    
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o-', 's-', '^-', 'd-']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for c in range(4):
        pur_list, pur_err_list = [], []
        for i in range(len(bins) - 1):
            in_bin = (var_array >= bins[i]) & (var_array < bins[i+1])
            
            is_c = (all_labels == c) & in_bin # Events that are in class c and fall into this bin
            pred_c = (all_preds == c) & in_bin # All events that the network classified as c (denominator)
            tp_c = is_c & pred_c # True positives (numerator)
            
            n_pred_c = np.sum(pred_c)
            W_pred_c = np.sum(w[pred_c]) # Sum the weights W to calculate all predicted c (denominator) and true positives (numerator)
            W_tp_c = np.sum(w[tp_c])
            
            if W_pred_c > 0 and n_pred_c > 0:
                pur = W_tp_c / W_pred_c # Purity calculation for the single bin
                pur_err = np.sqrt(max(pur * (1.0 - pur), 0.0) / n_pred_c) # Purity error (binomial)
            else:
                pur, pur_err = np.nan, 0.0
                
            pur_list.append(pur)
            pur_err_list.append(pur_err)
            
        ax.errorbar(bin_centers, pur_list, yerr=pur_err_list, fmt=markers[c], color=colors[c],
                    lw=2, capsize=3, label=f'Purity {CLASS_NAMES[c]}')
        
    ax.set_title(f'ParT: Channel Purities (Cross Section-weighted) vs {var_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Channel Purity (Cross Section-weighted)')
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_tau_contamination_vs_var(var_array, bins, var_name, xlabel, filename, use_weights=True):
    """
    Breaks down the false positives among the events classified as Tau (pred == 0),
    showing the contamination fraction of Muons, Electrons, and Hadrons in each bin.
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    w = sample_weights if use_weights else np.ones_like(all_labels, dtype=np.float32)
    
    bkg_classes = [1, 2, 3] # Consider only non-tau classes
    colors = {1: 'blue', 2: 'green', 3: 'orange'}
    markers = {1: 's-', 2: '^-', 3: 'd-'}
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for k in bkg_classes:
        cont_list, cont_err_list = [], []
        for i in range(len(bins) - 1):
            in_bin = (var_array >= bins[i]) & (var_array < bins[i+1])
            
            pred_tau = (all_preds == 0) & in_bin # All events predicted as tau (denominator) in the bin
            false_tau_from_k = (all_labels == k) & pred_tau # Events predicted as tau but which are actually something else (false positives)
            
            n_pred_tau = np.sum(pred_tau)
            W_pred_tau = np.sum(w[pred_tau])
            W_fp_k = np.sum(w[false_tau_from_k])
            
            if W_pred_tau > 0 and n_pred_tau > 0:
                cont = W_fp_k / W_pred_tau # Contamination fraction of class k (predicted taus that are actually k)
                cont_err = np.sqrt(max(cont * (1.0 - cont), 0.0) / n_pred_tau)
            else:
                cont, cont_err = np.nan, 0.0
                
            cont_list.append(cont)
            cont_err_list.append(cont_err)
            
        ax.errorbar(bin_centers, cont_list, yerr=cont_err_list, fmt=markers[k], color=colors[k],
                    lw=2, capsize=3, label=f'Contamination from {CLASS_NAMES[k]}')
        
    ax.set_title(f'ParT: Tau Background Contamination vs {var_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Contamination Fraction in Pred Tau')
    ax.set_ylim(bottom=0.0) # Automatic Y scaling at the top to clearly see even small contaminations
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


# Plot 4-channel Purity as a function of cos(theta_thrust) and Visible Energy
plot_all_channels_purity_vs_var(
    var_array=test_cos_theta,
    bins=cos_bins,
    var_name=r'$\cos(\theta_{\mathrm{thrust}})$',
    xlabel=r'$\cos(\theta_{\mathrm{thrust}})$',
    filename="part_all_purities_vs_costheta.png",
    use_weights=True
)

plot_all_channels_purity_vs_var(
    var_array=test_e_vis,
    bins=evis_bins,
    var_name='Visible Energy',
    xlabel=r'$E_{\mathrm{vis}}$ [GeV]',
    filename="part_all_purities_vs_evis.png",
    use_weights=True
)

# Plot specific Contamination in the Tau channel as a function of cos(theta_thrust) and Visible Energy
plot_tau_contamination_vs_var(
    var_array=test_cos_theta,
    bins=cos_bins,
    var_name=r'$\cos(\theta_{\mathrm{thrust}})$',
    xlabel=r'$\cos(\theta_{\mathrm{thrust}})$',
    filename="part_tau_contamination_vs_costheta.png",
    use_weights=True
)

plot_tau_contamination_vs_var(
    var_array=test_e_vis,
    bins=evis_bins,
    var_name='Visible Energy',
    xlabel=r'$E_{\mathrm{vis}}$ [GeV]',
    filename="part_tau_contamination_vs_evis.png",
    use_weights=True
)

# Distribution of the cos(theta_thrust) value

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['red', 'blue', 'green', 'orange']

# Unweighted left plot
for i in range(4): # Consider events that actually belong to class i
    mask = (all_labels == i)
    ax1.hist(test_cos_theta[mask], bins=cos_bins, histtype='step', lw=2, # Use the same bins as the efficiency and purity plots
             label=f'True {CLASS_NAMES[i]}', color=colors[i], density=False) # Density = False because I want the number of events

ax1.set_title(r'ParT: $\cos(\theta_{\mathrm{thrust}})$ (MC Statistics)')
ax1.set_xlabel(r'$\cos(\theta_{\mathrm{thrust}})$')
ax1.set_ylabel('Number of Events (Balanced)')
ax1.legend(loc='lower left')
ax1.grid(True, linestyle='--', alpha=0.6)

# Right plot weighted by cross-section
for i in range(4):
    mask = (all_labels == i)
    ax2.hist(test_cos_theta[mask], bins=cos_bins, weights=sample_weights[mask], 
             histtype='step', lw=2, label=f'True {CLASS_NAMES[i]}', color=colors[i], density=False)

ax2.set_title(r'ParT: $\cos(\theta_{\mathrm{thrust}})$ (Physics Expected)')
ax2.set_xlabel(r'$\cos(\theta_{\mathrm{thrust}})$')
ax2.set_ylabel('Expected Events (a.u., Cross Section-weighted)')
ax2.set_yscale('log')
ax2.legend(loc='lower left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.savefig(os.path.join(OUTPUT_DIR, "part_costheta_distributions.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n>>> Operation completed successfully!")
print(f"  - Best model saved to: {best_model_path}")
print(f"  - Plots saved in the folder '{OUTPUT_DIR}/'")

# ====================================================================
# END OF PART 4
# ====================================================================
