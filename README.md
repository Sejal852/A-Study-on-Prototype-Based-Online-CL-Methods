# A-Study-on-Prototype-Based-Online-CL-Methods
This project implements prototype-based online continual learning methods from my MSc thesis at University of Padova (2024-2025). It enhances replay-based CL with PDI. Prototype Distance Interpolation (PDI) is a technique used in this thesis to enhance continual learning by interpolating between class prototypes based on their distances.

## Overview
Continual Learning (CL) aims to enable models to learn from sequential data streams without forgetting prior knowledge. This project explores replay-based CL in an online setting (single-pass data processing), emphasizing prototype representations (mean embeddings) for efficient memory management and classification.
Key contributions:

**Adaptive Prototype Feedback (APF)**: A sampling-based mix-up technique that predicts misclassification risks by measuring distances between class prototypes in the memory buffer. This is integrated into baseline methods to prioritize "confused" classes and strengthen decision boundaries.
Enhanced Baselines:

**Prototype Distance Interpolation**: Prototype Distance Interpolation (PDI) is a technique used in this thesis to enhance continual learning by interpolating between class prototypes based on their distances. PDI focuses on creating new samples by linearly interpolating between prototypes of "confused" classes. This strengthens decision boundaries, improving model robustness.
Experiments on standard benchmarks (CIFAR-10 and CIFAR-100) demonstrate 3-5% accuracy improvements over baselines, with evaluations across varying memory sizes (e.g., 500, 2000).

### Enhanced Baselines:

Enhanced Experience Replay (ER): Improves vanilla ER by incorporating APF for dynamic replay sample selection and updates.
Vanilla CoPE + APF: Extends Continual Prototype Evolution (CoPE) with APF for better prototype evolution.
Enhanced CoPE + PDI: Further refines CoPE with Prototype Distance Integration (PDI) for misclassification-focused updates.


Experiments on standard benchmarks (CIFAR-10 and CIFAR-100) demonstrate 3-5% accuracy improvements over baselines, with evaluations across varying memory sizes (e.g., 500, 2000).

The code is implemented in Python using the Avalanche Library, with support for contrastive learning, nearest class mean classifiers, and hyperparameter tuning.

### Files

This repo includes 7 main Python files implementing the methods in Avalanche:

  
  Vanilla_ER.py: Vanilla Experience Replay (ER) baseline.
  
  Enhanced_er.py: Enhanced ER with adaptive sampling for better replay selection.
  
  Vanilla_CoPE.py: Vanilla Continual Prototype Evolution (CoPE).
  
  Vanilla_CoPE_APF.py: Vanilla CoPE integrated with Adaptive Prototype Feedback (APF).
  
  Enhanced_CoPE_PDI.py: Enhanced CoPE with Prototype Distance Interpolation (PDI) for misclassification-focused updates.
  
  OnPro_Baseline.py: The OnPro method runs under the settings from the original paper.
  
  OnPro_Standard.py: OnPro adapted to run under the same standardized settings as other methods (e.g., unified benchmarks, memory sizes).

### Features

  Modular CL frameworks for easy extension.
  
  Prototype management and mix-up augmentation.
  
  Evaluation metrics: Average accuracy, forgetting rate.
  
  Visualizations for performance analysis (e.g., accuracy vs. iterations, time comparisons).


The full thesis PDF is available here. For details on methodology, results, and related works, refer to the document.

