# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch implementation of the **Tolman-Eichenbaum Machine (TEM)** — a biologically-inspired neural network modeling hippocampal-entorhinal function for spatial memory and generalization. Based on Whittington et al. (2020), [biorxiv](https://www.biorxiv.org/content/10.1101/770495v2.full).

**Dependencies**: Python >= 3.6, PyTorch >= 1.6, numpy, scipy, matplotlib (no requirements.txt — install manually).

## Common Commands

```bash
# Train the model
python run.py

# Test/analyze a trained model (edit test.py to specify which run/checkpoint to load)
python test.py

# Train LSTM baseline for comparison
python run_lstm.py
```

There are no formal test suites, linters, or build steps. This is a research codebase.

## Architecture

### Core Representations
The TEM models two brain regions with three key variables:
- **`g`** — abstract location (entorhinal grid cells): list of tensors, one per frequency module
- **`p`** — grounded location (hippocampal place cells): list of tensors, one per frequency module
- **`x`** — sensory observations: what the agent currently sees

### Key Files
- **`model.py`** — `Model(nn.Module)`: the TEM neural network. `forward()` processes full walks; `iteration()` is one TEM timestep (transition → inference → generative → memory update). Also contains `MLP`, `Attractor` (pattern completion), and `Iteration` (data container for one step).
- **`parameters.py`** — all hyperparameters returned as a dict from `default_params()`. 5 hierarchical frequency modules, 20k training iterations, batch size 16, dynamic walk lengths (25–300 steps).
- **`world.py`** — `World` class loads JSON environment graphs and simulates agent navigation with action-based transitions.
- **`run.py`** — training loop: Adam optimizer, scheduled LR decay, TensorBoard logging, BPTT with rollout windows.
- **`test.py`** — loads a checkpoint and runs analysis; edit the date/run/checkpoint variables at the top to select which model to evaluate.
- **`analyse.py`** — post-hoc analysis: accuracy, zero-shot generalization, rate maps, baseline comparison.
- **`utils.py`** — activation functions (`softmax`, `normalise`, `relu`), loss functions (`squared_error`, `cross_entropy`), directory management.
- **`plot.py`** — visualization of weights, Hebbian memory matrices, and environment graphs.

### Environment Files
`envs/*.json` define graph-structured environments (4×4, 5×5, 10×10 grids). Each location has an id, observation, (x,y) coords, and action-transition adjacency. Optional shiny-object properties define rewarded locations with associated policies.

### Training Data Flow
1. `World` generates random walks through a graph environment.
2. At each step, the model receives an action + observation.
3. `Model.iteration()` runs: path integration updates `g`; inference network infers `g` and `p` from sensory input; Hebbian memory retrieves `p` from `g`; generative network predicts the next observation.
4. An 8-component loss (covering observations, grounded locations, abstract locations) is backpropagated over rollout windows.

### Checkpoint Structure
Training saves to `../Summaries/<date>/<run>/` with subdirectories for model checkpoints, TensorBoard logs, and parameter files. `test.py` loads from these paths.
