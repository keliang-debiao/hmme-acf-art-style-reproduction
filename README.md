# HMME-ACF Reproduction Code

This repository contains the reproduction code for **Hierarchical Multi-Model Ensemble with Auxiliary CLIP Fusion (HMME-ACF)** for fine-grained art style recognition and hierarchical diagnostic evaluation.

The repository includes code for model training, prediction generation, probability fusion, hierarchical aggregation, ablation analysis, component-complementarity diagnostics, efficiency measurement, statistical testing, feature-space visualization, cross-source evaluation, and distillation-oriented deployment experiments.

No datasets, trained weights, prediction files, intermediate features, logs, tables, figures, or experimental result files are included in this repository.

## Repository Contents

```text
.
├── configs/
│   └── default.yaml
├── requirements.txt
├── scripts/
│   ├── prepare_artbench_split.py
│   ├── train_visual_model.py
│   ├── train_clip_classifier.py
│   ├── predict_tta.py
│   ├── search_fusion_weights.py
│   ├── evaluate_fusion.py
│   ├── evaluate_hierarchical_aggregation.py
│   ├── train_hierarchical_multitask.py
│   ├── evaluate_hmme_lite.py
│   ├── measure_efficiency.py
│   ├── train_distilled_student.py
│   ├── analyze_component_complementarity.py
│   ├── analyze_clip_transitions.py
│   ├── analyze_seed_stability.py
│   ├── run_bootstrap_and_mcnemar.py
│   ├── analyze_feature_space_tsne.py
│   ├── evaluate_wikiart_subset.py
│   └── analyze_weight_sensitivity.py
└── src/
    └── hmme_acf/
        ├── __init__.py
        ├── config.py
        ├── datasets.py
        ├── transforms.py
        ├── models.py
        ├── clip_models.py
        ├── train.py
        ├── inference.py
        ├── fusion.py
        ├── hierarchy.py
        ├── metrics.py
        ├── statistics.py
        ├── complementarity.py
        ├── efficiency.py
        ├── distillation.py
        ├── feature_analysis.py
        └── utils.py
```

## Configuration Files

### `configs/default.yaml`

Contains the default experiment configuration, including dataset paths, class names, model names, training hyperparameters, test-time augmentation settings, probability-fusion settings, hierarchical label mappings, statistical-analysis settings, efficiency-measurement settings, and output-path placeholders.

### `requirements.txt`

Lists the Python dependencies required by the reproduction scripts, including deep-learning, image-processing, numerical-computing, statistical-analysis, and visualization libraries.

## Source Code Modules

### `src/hmme_acf/config.py`

Provides utilities for loading YAML configuration files, resolving paths, setting random seeds, and sharing experiment settings across scripts.

### `src/hmme_acf/datasets.py`

Defines dataset loaders for ArtBench-10-style directory structures and compatible external subsets. The module supports image-path indexing, class-label mapping, split loading, and metadata handling.

### `src/hmme_acf/transforms.py`

Contains image preprocessing and augmentation pipelines, including RGB conversion, resizing, center cropping, training augmentation, ImageNet normalization, and test-time augmentation views.

### `src/hmme_acf/models.py`

Defines visual backbone construction for the ImageNet-pretrained component classifiers used in the HMME-ACF framework.

### `src/hmme_acf/clip_models.py`

Contains code for constructing and fine-tuning the CLIP ViT-B/32 classifier used as a low-weight auxiliary component.

### `src/hmme_acf/train.py`

Provides reusable training and validation loops for visual classifiers, CLIP fine-tuning, hierarchical multitask training, and student-model training.

### `src/hmme_acf/inference.py`

Implements single-view and test-time-augmentation inference. It generates model-level probability outputs that can be consumed by the fusion and analysis scripts.

### `src/hmme_acf/fusion.py`

Implements validation-selected weighted probability fusion, equal-weight fusion, CLIP-augmented fusion, and HMME-Lite operating-point construction.

### `src/hmme_acf/hierarchy.py`

Defines deterministic mappings from the original fine-grained label space to the diagnostic hierarchical label spaces. It also provides utilities for post-inference probability aggregation and alternative-taxonomy evaluation.

### `src/hmme_acf/metrics.py`

Contains metric utilities for top-k accuracy, macro-F1, balanced accuracy, per-class precision, per-class recall, per-class F1-score, and confusion-matrix computation.

### `src/hmme_acf/statistics.py`

Provides statistical-analysis utilities, including bootstrap confidence intervals, paired contingency tables, McNemar testing, seed-level summaries, and helper functions for statistical reporting.

### `src/hmme_acf/complementarity.py`

Implements component-complementarity diagnostics, including disagreement rate, Q-statistic, error correlation, double-fault measure, oracle accuracy, and error-recovery analysis.

### `src/hmme_acf/efficiency.py`

Contains utilities for measuring per-image inference latency, model parameter counts, peak memory usage, and accuracy-efficiency operating points.

### `src/hmme_acf/distillation.py`

Provides utilities for distilling ensemble soft predictions into a lower-cost student model under a deployment-oriented experimental setting.

### `src/hmme_acf/feature_analysis.py`

Contains code for feature extraction, feature normalization, PCA preprocessing, t-SNE embedding generation, and feature-space structure diagnostics.

### `src/hmme_acf/utils.py`

Provides shared helper functions for logging, file I/O, device selection, reproducibility control, prediction serialization, and safe directory creation.

## Experiment Scripts

### `scripts/prepare_artbench_split.py`

Prepares or verifies the ArtBench-10 training, validation, and test split files expected by the remaining scripts.

### `scripts/train_visual_model.py`

Trains one visual component classifier under the common training protocol defined in the configuration file.

### `scripts/train_clip_classifier.py`

Fine-tunes the CLIP ViT-B/32 classifier used as the auxiliary probability source.

### `scripts/predict_tta.py`

Runs inference with single-view or 14-view test-time augmentation and writes model-level probability outputs.

### `scripts/search_fusion_weights.py`

Searches validation-selected probability-fusion weights for the visual ensemble and auxiliary CLIP component.

### `scripts/evaluate_fusion.py`

Evaluates fused 10-class probability predictions under the selected fusion configuration.

### `scripts/evaluate_hierarchical_aggregation.py`

Applies deterministic post-inference probability aggregation from the fine-grained label space to the diagnostic hierarchical label spaces.

### `scripts/train_hierarchical_multitask.py`

Trains the explicit hierarchical multitask comparator used to compare post-hoc aggregation with hierarchical supervision.

### `scripts/evaluate_hmme_lite.py`

Evaluates lower-cost HMME-Lite configurations based on subsets of the visual component models.

### `scripts/measure_efficiency.py`

Measures inference latency, parameter count, and memory-related quantities for selected operating points.

### `scripts/train_distilled_student.py`

Trains the deployment-oriented distilled student model using ensemble soft predictions.

### `scripts/analyze_component_complementarity.py`

Computes pairwise and aggregate complementarity diagnostics among the visual component classifiers.

### `scripts/analyze_clip_transitions.py`

Analyzes image-level prediction transitions induced by adding the auxiliary CLIP component to the visual ensemble.

### `scripts/analyze_seed_stability.py`

Summarizes repeated runs across random seeds using the metrics requested in the configuration file.

### `scripts/run_bootstrap_and_mcnemar.py`

Computes bootstrap confidence intervals and paired McNemar contingency statistics for prespecified model comparisons.

### `scripts/analyze_feature_space_tsne.py`

Extracts features, performs dimensionality reduction, generates t-SNE embeddings, and computes feature-space structure diagnostics.

### `scripts/evaluate_wikiart_subset.py`

Runs cross-source evaluation on a WikiArt-style subset with the same target label space.

### `scripts/analyze_weight_sensitivity.py`

Evaluates uniform-weight baselines and local perturbations around the validation-selected fusion weights.

## Not Included

This repository does not include:

- ArtBench-10 image files
- WikiArt image files
- trained model checkpoints
- ensemble prediction files
- CLIP prediction files
- extracted feature files
- generated t-SNE coordinates
- output tables
- output figures
- logs
- experimental result summaries
- manuscript files

