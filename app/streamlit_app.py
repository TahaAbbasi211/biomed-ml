import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomed.datasets import BonnEEGDataset  # noqa: E402
from biomed.inference import (load_bundle, parse_signal, predict_proba,  # noqa: E402
                              split_segments)

MODEL_PATH = ROOT / "models" / "eeg_random_forest.joblib"
RAW_DIR = ROOT / "data" / "eeg" / "raw"
CLASS_NAMES = {1: "Seizure", 2: "Tumor area", 3: "Healthy area",
               4: "Eyes closed", 5: "Eyes open"}

st.set_page_config(page_title="EEG seizure detector", page_icon="??")
st.title("EEG seizure detector")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run: "
             "python -m biomed.train configs/eeg_rf.yaml")
    st.stop()


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


@st.cache_data
def get_local_data():
    return BonnEEGDataset().load()


bundle = get_bundle()
seg_len = bundle["segment_len"]
st.caption(f"Input: segments of {seg_len} samples (about 1 second at "
           "173.61 Hz), the format of the Bonn dataset.")

source = st.radio("Signal source",
                  ["Upload a file", "Pick a sample from local Bonn data"])

segments, true_label = None, None

if source == "Upload a file":
    f = st.file_uploader("Text or CSV file with numbers", type=["txt", "csv"])
    if f is not None:
        try:
            values = parse_signal(f.read().decode("utf-8", errors="ignore"))
            segments = split_segments(values, seg_len)
        except ValueError as e:
            st.error(str(e))
else:
    if not RAW_DIR.exists():
        st.info("Local data not found at data/eeg/raw. Use the upload option.")
    else:
        X, y, _ = get_local_data()
        cls = st.selectbox("Class", list(CLASS_NAMES),
                           format_func=lambda c: CLASS_NAMES[c])
        pool = X[y == cls]
        i = st.slider("Sample index", 0, len(pool) - 1, 0)
        segments, true_label = pool[i:i + 1], CLASS_NAMES[cls]

if segments is not None:
    proba = predict_proba(bundle, segments)
    mean_p = float(proba.mean())

    st.subheader("Result")
    if mean_p >= 0.5:
        st.error(f"Seizure-like activity: probability {mean_p:.1%}")
    else:
        st.success(f"No seizure detected: seizure probability {mean_p:.1%}")
    if true_label:
        st.caption(f"True class of this sample: {true_label}")

    st.subheader("Signal (first segment)")
    st.line_chart(pd.DataFrame({"amplitude": segments[0]}))

    if len(segments) > 1:
        st.subheader("Per-segment probability")
        st.bar_chart(pd.DataFrame({"seizure probability": proba}))
