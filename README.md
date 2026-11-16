# biomed-ml

An extensible machine learning framework for biomedical data (signals and tabular).
Each dataset, feature extractor and model is a separate module, so adding a new one
means adding a single file.

Status: work in progress. The package scaffolding and the exploration notebooks exist;
the module implementations are next.

## Layout

```
src/biomed/      installable package (src layout, name: biomed)
  datasets/      dataset loaders (placeholder)
  features/      feature extractors (placeholder)
  models/        models (placeholder)
  evaluation/    evaluation helpers (placeholder)
  pipeline.py    end-to-end pipeline (placeholder)
notebooks/       exploration and modeling notebooks
  01_data_exploration.ipynb
  02_feature_exploration.ipynb
  03_modeling.ipynb
  04_evaluation.ipynb
tests/           pytest tests
```

## Setup

Python >= 3.9.

```bash
pip install -r requirements.txt
```

Dependencies: NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyYAML, pytest, Streamlit.

## What's next

- Implement the first dataset loader, feature extractor and baseline model modules
- Wire up `pipeline.py` end to end
- Add tests for the pipeline
