# Change: Add Model Interpretability and Analysis Tools

## Why
Understanding what models learn about rebracketing is critical for validating the taxonomy and discovering emergent patterns. Interpretability tools reveal which features drive predictions, how chromatic modes are represented, and whether models discover ontological structures beyond explicit training.

## What Changes
- Add attention visualization for text, audio, and cross-modal attention
- Add embedding space analysis tools (TSNE, UMAP projections)
- Add feature attribution via Integrated Gradients and SHAP
- Add counterfactual explanation generation
- Add activation analysis and layer-wise representations
- Implement chromatic geometry analysis in embedding space
- Add visualization utilities and interactive exploration tools

## Impact
- Affected specs: model-interpretability (new capability)
- Affected code:
  - `training/interpretability/` (new directory)
  - `training/visualization/` - plots and interactive visualizations
  - `training/analysis/` - embedding analysis, geometry
- Dependencies: captum (PyTorch interpretability), umap-learn, plotly
