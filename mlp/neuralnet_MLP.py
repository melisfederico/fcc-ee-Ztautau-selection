#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ====================================================================
# 1. PARAMETER CONFIGURATION
# ====================================================================

# Fix seed for reproducibility: identical weight initialization and data splitting at every run
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Angular cut
COS_THETA_CUT = 0.993 # cos(120millirad)=0.993. Also tested with COS_THETA_CUT = 0.7
INVALID_VAL = -999.0

# Network hyperparameters
EPOCHS = 30
BATCH_SIZE = 512 # Number of events processed simultaneously before updating weights
LEARNING_RATE = 0.001

INPUT_DIR = "output_root_files_MLP" # Folder containing the root files output from the analysis script
OUTPUT_DIR = "neuralnet_MLP_output_files_cos0993" # NB: must be changed if using cos07
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

# Input variables (10 features: including the 9 kinematic variables + cos_theta_thrust)
FEATURES = [
    "N_ch", "theta_acol", "theta_iso", 
    "E_rad", "P_rad", "E_tot_norm", 
    "Thrust_val", "p1_norm", "p2_norm",
    "cos_theta_thrust"
]

# Variables to extract from the root files
BRANCHES_TO_READ = FEATURES

# ====================================================================
# 2. DATA LOADING AND PREPROCESSING
# ====================================================================
print(">>> [1/5] Loading ROOT files and extracting data...")

# Create dictionaries to collect DataFrames and count pre/post-filter events for each class
dfs_by_class = {0: [], 1: [], 2: [], 3: []}
total_initial_by_class = {0: 0, 1: 0, 2: 0, 3: 0}
total_selected_by_class = {0: 0, 1: 0, 2: 0, 3: 0}

# Open each root file and extract the required columns (BRANCHES_TO_READ)
for file_name, label in PROCESSES.items():
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Open the 'events' TTree (event table) and extract them as a Pandas DataFrame
    with uproot.open(file_path) as file:
        tree = file["events"]
        df_temp = tree.arrays(BRANCHES_TO_READ, library="pd") # Reads the 10 selected columns
        
        # Filter 1: geometric angular cut (cos_theta_thrust is already in absolute value in the root file)
        mask_fiducial = df_temp["cos_theta_thrust"] < COS_THETA_CUT
        n_init = int(mask_fiducial.sum()) # Events in the fiducial volume before discarding those with -999
        
        # Filter 2: remove non-physical values (empty hemispheres return -999)
        mask_valid = (df_temp["theta_acol"] != INVALID_VAL) & (df_temp["theta_iso"] != INVALID_VAL)
        
        # Combine the two masks keeping only events that satisfy both 
        df_clean = df_temp[mask_fiducial & mask_valid].copy()
        
        # Assign a target to the surviving events
        df_clean["target"] = label # Create a column with the label (0->tau, 1->mu, 2->e, 3->quark)
        
        # Append the DataFrame to the list corresponding to its class and update counts
        dfs_by_class[label].append(df_clean)
        total_initial_by_class[label] += n_init
        total_selected_by_class[label] += len(df_clean)

        # Print how many events survived the filter compared to those within the angular cut 
        print(f"  - {file_name}: {len(df_clean)}/{n_init} events selected ({len(df_clean)/n_init*100:.2f}%).")

# Calculate the fraction of events surviving the cut for each of the 4 classes (preselection efficiency)
presel_eff = np.array([
    total_selected_by_class[c] / total_initial_by_class[c] for c in range(4)
], dtype=np.float32)

# ====================================================================
# 3. CLASS BALANCING
# ====================================================================
print("\n>>> Class balancing in progress...")

# Ensure all classes have the same number of events (the issue arises from quarks, which are more numerous)

# Merge fragments for each class (combine the 4 hadronic files into a single class labeled 3)
for label in dfs_by_class:
    dfs_by_class[label] = pd.concat(dfs_by_class[label], ignore_index=True)
    print(f"  - Total events ready for Class {label}: {len(dfs_by_class[label])}")

# Find the minimum number of events across the 4 classes
min_events = min(len(df) for df in dfs_by_class.values())
print(f"\n  -> Cutting all classes to exactly {min_events} events for 100% balance.")

# Extract 'min_events' from each class and create the final dataset
balanced_dfs = []
for label, df in dfs_by_class.items():
    df_sampled = df.sample(n=min_events, random_state=RANDOM_SEED).copy() # Random sampling
    balanced_dfs.append(df_sampled)

# Concatenate all balanced samples and shuffle the entire dataset
df_all = pd.concat(balanced_dfs, ignore_index=True)
df_all = df_all.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True) # Randomly shuffle the samples
df_all = df_all.dropna() # Remove any residual NaNs

X_numpy = df_all[FEATURES].values.astype(np.float32)
y_numpy = df_all["target"].values

# Save the physical (unstandardized) vectors of cos(theta_thrust) and visible energy (in GeV, multiplying E_tot_norm by sqrts=91.2)
cos_theta_numpy = df_all["cos_theta_thrust"].values.astype(np.float32)
e_vis_numpy = (df_all["E_tot_norm"].values * 91.2).astype(np.float32)

# ====================================================================
# 4. SPLITTING AND STANDARDIZATION
# ====================================================================
print("\n>>> [2/5] Preparing Train/Validation/Test sets...")

# Split by indices into Train (70%), Temporary (30%)
indices = np.arange(len(y_numpy))
idx_train, idx_temp, y_train, y_temp = train_test_split(
    indices, y_numpy, test_size=0.30, random_state=RANDOM_SEED, stratify=y_numpy # stratify=y_numpy ensures the 25% proportion per class is maintained in each set
)

# Split Temporary into Validation (15%) and Test (15%)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

X_train = X_numpy[idx_train]
X_val   = X_numpy[idx_val]
X_test  = X_numpy[idx_test]

# Standardization (ONLY on the train set): transform each variable by subtracting its mean and dividing by the std dev
# Standardize all variables to mean 0 and variance 1 so that gradient descent is balanced
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Conversion to PyTorch tensors
tensor_X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.long)
tensor_X_val   = torch.tensor(X_val_scaled, dtype=torch.float32)
tensor_y_val   = torch.tensor(y_val, dtype=torch.long)
tensor_X_test  = torch.tensor(X_test_scaled, dtype=torch.float32)
tensor_y_test  = torch.tensor(y_test, dtype=torch.long)

# Data flow management during training.
# shuffle=True only for the train set: randomly shuffles event order at the start of each epoch (not needed for test and val)
train_loader = DataLoader(TensorDataset(tensor_X_train, tensor_y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(tensor_X_val, tensor_y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(tensor_X_test, tensor_y_test), batch_size=BATCH_SIZE, shuffle=False)

# Print a summary of the number of events 
print(f"  - Training Set:   {len(X_train)} events")
print(f"  - Validation Set: {len(X_val)} events")
print(f"  - Test Set:       {len(X_test)} events")

# ====================================================================
# 5. PYTORCH MODEL DEFINITION
# ====================================================================

class TauClassifier(nn.Module):
    def __init__(self, input_dim=9, num_classes=4):
        super(TauClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), # Normalize the 512 intermediate activation values of the batch and force them to have mean 0 and variance 1 before passing to the next layer
            nn.ReLU(),
            nn.Dropout(0.20), # Randomly drop 20% of the neurons. This forces the network not to rely on a single feature, preventing overfitting
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            
            nn.Linear(64, num_classes) # Outputs 4 continuous numerical scores (logits)
        )

    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TauClassifier(input_dim=len(FEATURES), num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ====================================================================
# 6. TRAINING LOOP
# ====================================================================
print("\n>>> [3/5] Starting neural network training...")

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}


best_val_loss = float('inf') # Initialize to infinity so that at the end of the first epoch the program saves the weights to compare with subsequent ones
best_model_path = os.path.join(OUTPUT_DIR, "mlp_tau_model_best.pth")

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() # Zero the gradients accumulated in the previous batch
        outputs = model(inputs) # Compute logits (scores)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0) # Sum mean error by 512 (batch size) to get the total epoch error  
        _, predicted = torch.max(outputs.data, 1) # Returns the maximum logit value and the corresponding index (0,1,2,3)
        total += labels.size(0) # Keep track of the total events seen so far
        correct += (predicted == labels).sum().item() # Count how many events out of the 512 were correctly predicted
        
    epoch_train_loss = running_loss / total
    epoch_train_acc = correct / total
    
    # --- VALIDATION PHASE ---
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad(): # No need to compute gradients
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = correct_val / total_val
    
    # Save metrics
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)
    
    # Save the best model: if epoch_val_loss is the lowest recorded so far, overwrite best_model_path
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), best_model_path)
        best_saved_epoch = epoch + 1
    
    if (epoch+1) % 5 == 0 or epoch == 0:  # Print progress to screen every 5 epochs
        print(f"  Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

# Save the weights of the last epoch as a backup 
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mlp_tau_model_last.pth"))
print(f"  -> Best model (Epoch {best_saved_epoch}) saved with Val Loss: {best_val_loss:.4f}")

# ====================================================================
# 7. TEST SET EVALUATION
# ====================================================================
print("\n>>> [4/5] Evaluating on the Test Set...")

# Use the best weights found
model.load_state_dict(torch.load(best_model_path))

model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # Softmax to get probabilities between 0 and 1
        probs = torch.softmax(outputs, dim=1) # Convert the 4 output logits into a normalized probability distribution between 0 and 1
        _, predicted = torch.max(outputs.data, 1) # Select the index of the class with the highest probability (final prediction)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Save the 2D array of all probabilities assigned to all test events
        # The first column contains the probability associated with the tau signal (needed for efficiency and purity plots)
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds) # Convert to numpy arrays
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Extraction of cos(theta) and visible energy (in GeV) for the test set events
test_cos_theta = cos_theta_numpy[idx_test]
test_e_vis = e_vis_numpy[idx_test]

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

# ====================================================================
# 8. PLOT GENERATION AND SAVING
# ====================================================================
print("\n>>> [5/5] Generating plots...")

# Learning curves (epoch-by-epoch trend of Loss (left) and Accuracy (right))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train Loss', color='blue', lw=2)
ax1.plot(history['val_loss'], label='Validation Loss', color='orange', lw=2)
ax1.set_title('MLP: Loss Function Evolution')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cross Entropy Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(history['train_acc'], label='Train Accuracy', color='blue', lw=2)
ax2.plot(history['val_acc'], label='Validation Accuracy', color='orange', lw=2)
ax2.set_title('MLP: Overall Accuracy Evolution')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "mlp_learning_curves.png"), dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrix normalized by TRUE LABEL (Efficiency) weighted by cross-section
cm_true = confusion_matrix(all_labels, all_preds, sample_weight=sample_weights, normalize='true')
disp_true = ConfusionMatrixDisplay(cm_true, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp_true.plot(cmap=plt.cm.Blues, ax=ax, values_format='.3f')
plt.title('MLP: Confusion Matrix Norm. by True Label (Efficiency) - Cross-Section Weighted')
plt.savefig(os.path.join(OUTPUT_DIR, "mlp_cm_weighted.png"), dpi=300, bbox_inches='tight')
plt.close()

# Efficiency confusion matrix (including discarded preprocessing events in the denominator)
cm_true_abs = cm_true * presel_eff[:, np.newaxis]
disp_abs = ConfusionMatrixDisplay(cm_true_abs, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp_abs.plot(cmap=plt.cm.Blues, ax=ax, values_format='.3f')
plt.title('MLP: Confusion Matrix Norm. by True Label (Efficiency) - incl. Discarded Events')
plt.savefig(os.path.join(OUTPUT_DIR, "mlp_cm_incl_discarded.png"), dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrix normalized by PREDICTED LABEL (Purity) weighted by cross-section
cm_pred = confusion_matrix(all_labels, all_preds, sample_weight=sample_weights, normalize='pred')
disp_pred = ConfusionMatrixDisplay(cm_pred, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 6))
disp_pred.plot(cmap=plt.cm.Oranges, ax=ax, values_format='.3f') 
plt.title('MLP: Confusion Matrix Norm. by Pred. Label (Purity) - Cross-Section Weighted')
plt.savefig(os.path.join(OUTPUT_DIR, "mlp_cm_pred_weighted.png"), dpi=300, bbox_inches='tight')
plt.close()

# Histogram of the probability of being a tau
tau_scores = all_probs[:, 0] # Probability associated with node 0 (tau)
fig, ax = plt.subplots(figsize=(9, 6))
colors = ['red', 'blue', 'green', 'orange']
for i in range(4):
    mask = (all_labels == i)
    ax.hist(tau_scores[mask], bins=50, histtype='step', lw=2, 
            label=f'True {CLASS_NAMES[i]}', color=colors[i], density=True)
ax.set_title('MLP Response: Tau Output Probability')
ax.set_xlabel('P(Tau)')
ax.set_ylabel('Normalized Density (A.U.)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "mlp_nn_score_response.png"), dpi=300, bbox_inches='tight')
plt.close()

# Efficiency and purity vs threshold (cross-section weighted)
thresholds = np.linspace(0.01, 0.99, 100)
effs, purs = [], []
total_tau_weighted = np.sum(sample_weights[all_labels == 0])

for t in thresholds:
    sel = (tau_scores >= t)
    TP = np.sum(sample_weights[sel & (all_labels == 0)])
    FP = np.sum(sample_weights[sel & (all_labels != 0)])
    effs.append(TP / total_tau_weighted if total_tau_weighted > 0 else 0)
    purs.append(TP / (TP + FP) if (TP + FP) > 0 else 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Efficiency and purity as a function of the threshold (left)
ax1.plot(thresholds, effs, label='Efficiency', color='green', lw=2)
ax1.plot(thresholds, purs, label='Purity', color='purple', lw=2)
ax1.set_title('MLP: Performance vs Decision Threshold')
ax1.set_xlabel('Decision Threshold on P(Tau)')
ax1.set_ylabel('Efficiency / Purity')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# ROC curve (purity vs efficiency) (right)
ax2.plot(effs, purs, color='darkred', lw=2)
ax2.set_title('MLP: Purity vs Efficiency Curve')
ax2.set_xlabel('Efficiency')
ax2.set_ylabel('Purity')
ax2.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, "mlp_efficiency_purity_roc.png"), dpi=300, bbox_inches='tight')
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
        in_bin = (var_array >= bins[i]) & (var_array < bins[i+1])
        
        is_tau = (all_labels == 0) & in_bin
        pred_tau = (all_preds == 0) & in_bin
        tp_mask = is_tau & pred_tau
        
        n_true_tau = np.sum(is_tau)
        n_pred_tau = np.sum(pred_tau)
        
        W_true_tau = np.sum(w[is_tau])
        W_pred_tau = np.sum(w[pred_tau])
        W_tp = np.sum(w[tp_mask])
        
        if W_true_tau > 0 and n_true_tau > 0:
            eff = W_tp / W_true_tau
            eff_err = np.sqrt(max(eff * (1.0 - eff), 0.0) / n_true_tau)
        else:
            eff, eff_err = np.nan, 0.0
            
        if W_pred_tau > 0 and n_pred_tau > 0:
            pur = W_tp / W_pred_tau
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
    
    ax.set_title(f'MLP: Tau Efficiency & Purity vs {var_name}')
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
    filename="mlp_eff_pur_vs_costheta.png",
    use_weights=True
)

# Plot as a function of Visible Energy (20 bins between 0 and 100 GeV at ECM=91 GeV)
evis_bins = np.linspace(0.0, 100.0, 21)
plot_eff_pur_vs_var(
    var_array=test_e_vis,
    bins=evis_bins,
    var_name='Visible Energy',
    xlabel=r'$E_{\mathrm{vis}}$ [GeV]',
    filename="mlp_eff_pur_vs_evis.png",
    use_weights=True
)

# Purity for individual channels and tau signal contamination

def plot_all_channels_purity_vs_var(var_array, bins, var_name, xlabel, filename, use_weights=True):
    """
    Calculates and plots the Purity of all 4 channels (Tau, Muon, Electron, Hadron)
    as a function of a kinematic variable to see where each channel loses sensitivity.
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    w = sample_weights if use_weights else np.ones_like(all_labels, dtype=np.float32)
    
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o-', 's-', '^-', 'd-']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for c in range(4):
        pur_list, pur_err_list = [], []
        for i in range(len(bins) - 1):
            in_bin = (var_array >= bins[i]) & (var_array < bins[i+1])
            
            is_c = (all_labels == c) & in_bin
            pred_c = (all_preds == c) & in_bin
            tp_c = is_c & pred_c
            
            n_pred_c = np.sum(pred_c)
            W_pred_c = np.sum(w[pred_c])
            W_tp_c = np.sum(w[tp_c])
            
            if W_pred_c > 0 and n_pred_c > 0:
                pur = W_tp_c / W_pred_c
                pur_err = np.sqrt(max(pur * (1.0 - pur), 0.0) / n_pred_c)
            else:
                pur, pur_err = np.nan, 0.0
                
            pur_list.append(pur)
            pur_err_list.append(pur_err)
            
        ax.errorbar(bin_centers, pur_list, yerr=pur_err_list, fmt=markers[c], color=colors[c],
                    lw=2, capsize=3, label=f'Purity {CLASS_NAMES[c]}')
        
    ax.set_title(f'MLP: Channel Purities (Cross Section-weighted) vs {var_name}')
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
    
    bkg_classes = [1, 2, 3]
    colors = {1: 'blue', 2: 'green', 3: 'orange'}
    markers = {1: 's-', 2: '^-', 3: 'd-'}
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for k in bkg_classes:
        cont_list, cont_err_list = [], []
        for i in range(len(bins) - 1):
            in_bin = (var_array >= bins[i]) & (var_array < bins[i+1])
            
            pred_tau = (all_preds == 0) & in_bin
            false_tau_from_k = (all_labels == k) & pred_tau
            
            n_pred_tau = np.sum(pred_tau)
            W_pred_tau = np.sum(w[pred_tau])
            W_fp_k = np.sum(w[false_tau_from_k])
            
            if W_pred_tau > 0 and n_pred_tau > 0:
                cont = W_fp_k / W_pred_tau
                cont_err = np.sqrt(max(cont * (1.0 - cont), 0.0) / n_pred_tau)
            else:
                cont, cont_err = np.nan, 0.0
                
            cont_list.append(cont)
            cont_err_list.append(cont_err)
            
        ax.errorbar(bin_centers, cont_list, yerr=cont_err_list, fmt=markers[k], color=colors[k],
                    lw=2, capsize=3, label=f'Contamination from {CLASS_NAMES[k]}')
        
    ax.set_title(f'MLP: Tau Background Contamination vs {var_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Contamination Fraction in Pred Tau')
    ax.set_ylim(bottom=0.0)
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
    filename="mlp_all_purities_vs_costheta.png",
    use_weights=True
)

plot_all_channels_purity_vs_var(
    var_array=test_e_vis,
    bins=evis_bins,
    var_name='Visible Energy',
    xlabel=r'$E_{\mathrm{vis}}$ [GeV]',
    filename="mlp_all_purities_vs_evis.png",
    use_weights=True
)

# Plot specific Contamination in the Tau channel as a function of cos(theta_thrust) and Visible Energy
plot_tau_contamination_vs_var(
    var_array=test_cos_theta,
    bins=cos_bins,
    var_name=r'$\cos(\theta_{\mathrm{thrust}})$',
    xlabel=r'$\cos(\theta_{\mathrm{thrust}})$',
    filename="mlp_tau_contamination_vs_costheta.png",
    use_weights=True
)

plot_tau_contamination_vs_var(
    var_array=test_e_vis,
    bins=evis_bins,
    var_name='Visible Energy',
    xlabel=r'$E_{\mathrm{vis}}$ [GeV]',
    filename="mlp_tau_contamination_vs_evis.png",
    use_weights=True
)

# Distribution of the cos(theta_thrust) value
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['red', 'blue', 'green', 'orange']

# Unweighted left plot
for i in range(4):
    mask = (all_labels == i)
    ax1.hist(test_cos_theta[mask], bins=cos_bins, histtype='step', lw=2,
             label=f'True {CLASS_NAMES[i]}', color=colors[i], density=False)

ax1.set_title(r'MLP: $\cos(\theta_{\mathrm{thrust}})$ (MC Statistics)')
ax1.set_xlabel(r'$\cos(\theta_{\mathrm{thrust}})$')
ax1.set_ylabel('Number of Events (Balanced)')
ax1.legend(loc='lower left')
ax1.grid(True, linestyle='--', alpha=0.6)

# Right plot weighted by cross-section
for i in range(4):
    mask = (all_labels == i)
    ax2.hist(test_cos_theta[mask], bins=cos_bins, weights=sample_weights[mask], 
             histtype='step', lw=2, label=f'True {CLASS_NAMES[i]}', color=colors[i], density=False)

ax2.set_title(r'MLP: $\cos(\theta_{\mathrm{thrust}})$ (Physics Expected)')
ax2.set_xlabel(r'$\cos(\theta_{\mathrm{thrust}})$')
ax2.set_ylabel('Expected Events (a.u., Cross Section-weighted)')
ax2.set_yscale('log')
ax2.legend(loc='lower left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.savefig(os.path.join(OUTPUT_DIR, "mlp_costheta_distributions.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n>>> Operation completed successfully!")
print(f"  - Best model saved to: {best_model_path}")
print(f"  - Plots saved in the folder '{OUTPUT_DIR}/'")
