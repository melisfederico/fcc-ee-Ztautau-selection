# FCC-ee Z -> tau tau Selection

This repository contains the data analysis and machine learning training scripts for the selection of $Z \to \tau^+\tau^-$ events at the FCC-ee IDEA detector. 

The project implements two different neural network architectures for signal-background discrimination: a Multi-Layer Perceptron (MLP) and a Particle Transformer (ParT).

## Repository Structure
* `analysis_MLP.py`: Extracts kinematic features and variables from ROOT EDM4hep files for the MLP network.
* `neuralnet_MLP.py`: PyTorch implementation of the MLP classifier.
* `analysis_ParT.py`: Extracts 4-momenta and particle attributes for the ParT network.
* `neuralnet_ParT.py`: PyTorch implementation of the Particle Transformer classifier.

## Requirements
* ROOT (with RDataFrame support)
* Python 3.x
* PyTorch
* uproot, awkward, pandas, numpy, scikit-learn, matplotlib

## Credits and Acknowledgements
The implementation of the Particle Transformer (`neuralnet_ParT.py`) relies on and modifies code from the following sources:
* **Particle Transformer Architecture**: Qu, H., Li, C., & Qian, S. (2022). *Particle Transformer for Jet Tagging*. Proceedings of the 39th International Conference on Machine Learning. [arXiv:2202.03772](https://arxiv.org/abs/2202.03772).
* **Original ParT Implementation**: Code from the [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer) repository (MIT License).
* **Tau-specific ParT Adaptation**: Code from the [HEP-KBFI/ml-tau-model](https://github.com/HEP-KBFI/ml-tau-model) repository (GPLv3 License).

## License
This project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.
