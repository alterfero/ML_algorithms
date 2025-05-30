"""
Online SVM demo with camera images
----------------------------------

pip install streamlit scikit-learn pillow
"""

import streamlit as st
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError

st.set_page_config(page_title="Continuous-Learning SVM", page_icon="🤖")

# ────────────────────────────── Session state ────────────────────────────── #
if "features" not in st.session_state:
    st.session_state.features: list[list[float]] = []
    st.session_state.labels:   list[str]         = []
    st.session_state.model                      = None  # will hold the SVC

# ────────────────────────────── UI – capture ────────────────────────────── #
st.title("Continuous-Learning SVM (mean RGB features)")

image_file = st.camera_input("1️⃣  Take or upload a picture", key="cam")

if image_file is not None:
    # ───── Feature extraction: mean R, G, B ───── #
    img = Image.open(image_file).convert("RGB")
    pixels = np.asarray(img).reshape(-1, 3).astype(np.float32)
    mean_rgb = pixels.mean(axis=0).tolist()  # 3-element feature vector

    st.image(img, caption="Captured frame", use_container_width=True)
    st.write(f"Feature vector (mean R,G,B): `{[round(x, 2) for x in mean_rgb]}`")

    # ───── Obtain / confirm label ───── #
    if st.session_state.model is None:
        # We are still collecting the first 5 samples
        label = st.text_input("2️⃣  Enter a label for this picture")
    else:
        # Model exists – predict then ask for confirmation
        try:
            prediction = st.session_state.model.predict([mean_rgb])[0]
        except NotFittedError:
            prediction = "(model not fitted)"
        st.markdown(f"**Predicted label:** `{prediction}`")
        correct = st.radio("Is this correct?", ("Yes", "No"), horizontal=True)
        label = prediction if correct == "Yes" else st.text_input("Enter correct label")

    # ───── Add to dataset & (re)train ───── #
    if st.button("💾  Add sample / retrain"):
        if label.strip() == "":
            st.warning("Please supply a non-empty label.")
            st.stop()

        st.session_state.features.append(mean_rgb)
        st.session_state.labels.append(label.strip())
        st.success(f"Sample added. Dataset now contains {len(st.session_state.labels)} samples.")

        # Train only when we have ≥5 samples *and* at least two distinct classes
        if len(st.session_state.labels) >= 5 and len(set(st.session_state.labels)) >= 2:
            st.info("Training SVM …")
            clf = SVC(kernel="linear", probability=False)
            clf.fit(st.session_state.features, st.session_state.labels)
            st.session_state.model = clf
            st.success(f"SVM trained on {len(st.session_state.labels)} samples "
                       f"({len(set(st.session_state.labels))} classes).")
        elif st.session_state.model is None:
            st.info("Need at least 5 samples and 2 classes before first training.")

# ────────────────────────────── Sidebar – dataset view ────────────────────────────── #
with st.sidebar:
    st.header("📊  Dataset")
    st.write(f"Total samples: **{len(st.session_state.labels)}**")
    if st.session_state.labels:
        st.table(
            {
                "R̄": [round(x[0], 1) for x in st.session_state.features],
                "Ḡ": [round(x[1], 1) for x in st.session_state.features],
                "B̄": [round(x[2], 1) for x in st.session_state.features],
                "Label": st.session_state.labels,
            }
        )
