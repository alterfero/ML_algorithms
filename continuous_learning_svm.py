import streamlit as st
import numpy as np
from PIL import Image
import io


def compute_features(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # average R, G, B
    return arr.mean(axis=(0, 1))


def train_binary_svm(X, y, epochs=100, lr=0.01, C=1.0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float32)
    b = 0.0
    for _ in range(epochs):
        for idx in range(n_samples):
            x_i = X[idx]
            y_i = y[idx]
            condition = y_i * (np.dot(x_i, w) + b) >= 1
            if condition:
                w -= lr * (2 / epochs * w)
            else:
                w -= lr * (2 / epochs * w - C * y_i * x_i)
                b -= lr * (-C * y_i)
    return w, b


def train_one_vs_rest(X, y, num_classes, epochs=100, lr=0.01, C=1.0):
    weights = []
    biases = []
    for cls in range(num_classes):
        y_binary = np.where(y == cls, 1, -1)
        w, b = train_binary_svm(X, y_binary, epochs=epochs, lr=lr, C=C)
        weights.append(w)
        biases.append(b)
    return np.array(weights), np.array(biases)


def predict(X, weights, biases):
    scores = X @ weights.T + biases
    return np.argmax(scores, axis=1)


st.title("Continuous Learning SVM Demo")

st.markdown(
    r"""
    ### How it works
    Each picture is represented by the average **red**, **green** and **blue** values.
    A linear Support Vector Machine (SVM) is trained using the hinge loss
    $$L = \frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i (w^T x_i + b))$$
    and updated with a simple sub-gradient descent.
    After the first five labelled images, the model predicts new images and
    updates itself with your feedback.
    """
)

# Initialize session state
for key, default in [
    ("features", []),
    ("labels", []),
    ("label_to_idx", {}),
    ("weights", None),
    ("biases", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def label_to_index(label_text):
    mapping = st.session_state["label_to_idx"]
    if label_text not in mapping:
        mapping[label_text] = len(mapping)
    return mapping[label_text]


def index_to_label(index):
    for lbl, idx in st.session_state["label_to_idx"].items():
        if idx == index:
            return lbl
    return str(index)


st.write("Samples so far:", len(st.session_state["features"]))

photo = st.camera_input("Take a picture")

if photo is not None:
    features = compute_features(photo.getvalue())

    if len(st.session_state["features"]) < 5:
        label_text = st.text_input("Label this picture")
        if st.button("Add to Dataset") and label_text:
            idx = label_to_index(label_text)
            st.session_state["features"].append(features)
            st.session_state["labels"].append(idx)
            st.experimental_rerun()
    else:
        if st.session_state["weights"] is not None:
            X = features[np.newaxis, :]
            pred_idx = predict(X, st.session_state["weights"], st.session_state["biases"])[0]
            pred_label = index_to_label(pred_idx)
            st.write("Model prediction:", pred_label)
            feedback = st.radio("Is this correct?", ("Yes", "No"))
            if feedback == "Yes":
                final_idx = pred_idx
            else:
                correct_text = st.text_input("Correct label")
                if not correct_text:
                    st.stop()
                final_idx = label_to_index(correct_text)
            if st.button("Add & Retrain"):
                st.session_state["features"].append(features)
                st.session_state["labels"].append(final_idx)
                X_all = np.array(st.session_state["features"], dtype=np.float32)
                y_all = np.array(st.session_state["labels"], dtype=np.int32)
                num_cls = len(st.session_state["label_to_idx"])
                w, b = train_one_vs_rest(X_all, y_all, num_cls)
                st.session_state["weights"] = w
                st.session_state["biases"] = b
                st.experimental_rerun()
        else:
            st.info("Not enough data to train. Label this picture to continue.")
            label_text = st.text_input("Label this picture")
            if st.button("Add to Dataset") and label_text:
                idx = label_to_index(label_text)
                st.session_state["features"].append(features)
                st.session_state["labels"].append(idx)
                if len(st.session_state["features"]) >= 5:
                    X_all = np.array(st.session_state["features"], dtype=np.float32)
                    y_all = np.array(st.session_state["labels"], dtype=np.int32)
                    num_cls = len(st.session_state["label_to_idx"])
                    w, b = train_one_vs_rest(X_all, y_all, num_cls)
                    st.session_state["weights"] = w
                    st.session_state["biases"] = b
                st.experimental_rerun()
