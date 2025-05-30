import streamlit as st
import numpy as np
import io
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pydub import AudioSegment

st.set_page_config(layout="wide")

# ---- PARAMETERS ----
N_BINS = 16
FREQ_MIN = 50
FREQ_MAX = 5000
SAMPLES_PER_RECORDING = 100
SAMPLE_RATE = 16000

# ---- SESSION STATE INIT ----
if "sounds" not in st.session_state:
    st.session_state["sounds"] = []
if "labels" not in st.session_state:
    st.session_state["labels"] = []
if "model" not in st.session_state:
    st.session_state["model"] = None
if "label_names" not in st.session_state:
    st.session_state["label_names"] = []
if "feature_data" not in st.session_state:
    st.session_state["feature_data"] = []
if "feature_labels" not in st.session_state:
    st.session_state["feature_labels"] = []
if "network_config" not in st.session_state:
    st.session_state["network_config"] = {"layers": 2, "neurons": 32}

# ---- FUNCTIONS ----

def extract_features(audio_file):
    # audio_file is an UploadedFile (file-like), not bytes!
    try:
        # Just pass it directly to librosa.load!
        audio_file.seek(0)  # rewind just in case
        wav, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        st.error(f"Could not load audio file: {e}")
        return None
    length = len(wav)
    samples = []
    for i in range(SAMPLES_PER_RECORDING):
        start = int(i * length / SAMPLES_PER_RECORDING)
        end = int((i + 1) * length / SAMPLES_PER_RECORDING)
        segment = wav[start:end]
        if len(segment) < 1:
            segment = np.zeros((int(length / SAMPLES_PER_RECORDING),))
        fft = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1 / sr)
        bins = np.zeros(N_BINS)
        bin_edges = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_BINS + 1)
        for j in range(N_BINS):
            idx = np.where((freqs >= bin_edges[j]) & (freqs < bin_edges[j+1]))[0]
            bins[j] = np.sum(fft[idx])
        if np.sum(bins) > 0:
            bins = bins / np.sum(bins)
        samples.append(bins)
    return np.array(samples)



def build_model(n_hidden_layers, n_neurons):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(N_BINS,)))
    for _ in range(n_hidden_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def visualize_network(model):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Only supports Sequential models
    layer_sizes = [N_BINS]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_sizes.append(layer.units)
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(len(layer_sizes) - 1)
    # Draw nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing, layer_top - m * v_spacing), v_spacing/4, color='skyblue', ec='black')
            ax.add_artist(circle)
    # Draw weights
    for n in range(len(layer_sizes) - 1):
        weights = model.layers[n].get_weights()[0] if hasattr(model.layers[n], 'get_weights') and len(model.layers[n].get_weights()) > 0 else None
        for i in range(layer_sizes[n]):
            for j in range(layer_sizes[n + 1]):
                lw = 0.2
                color = 'gray'
                if weights is not None:
                    weight = weights[i, j]
                    lw = min(1, 0.2 + abs(weight))
                    color = 'red' if weight > 0 else 'blue'
                ax.plot([n * h_spacing, (n + 1) * h_spacing],
                        [(v_spacing * (layer_sizes[n] - 1) / 2.0) - i * v_spacing,
                         (v_spacing * (layer_sizes[n + 1] - 1) / 2.0) - j * v_spacing],
                        lw=lw, color=color, alpha=0.6)
    ax.axis('off')
    plt.tight_layout()
    return fig

def onehot(label):
    return [1, 0] if label == 0 else [0, 1]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ---- MAIN APP ----

st.title("🎤 Neural Network Sound Classifier Demo")

# ---- Step 1: Recording ----
st.header("1. Record a Sound")
with st.form(key="audio_form"):
    audio_bytes = st.audio_input("Record or upload a 10-second sound")
    label = st.text_input("Label for this sound", "")
    submit = st.form_submit_button("Save this recording")
    if submit and audio_bytes is not None and label.strip() != "":
        st.session_state["sounds"].append(audio_bytes)
        st.session_state["labels"].append(label.strip())
        if label.strip() not in st.session_state["label_names"]:
            st.session_state["label_names"].append(label.strip())
        st.success(f"Saved sound with label: {label.strip()}")

# Display the collected data
st.write(f"Collected {len(st.session_state['sounds'])} sounds. Need at least 2 for training.")

# ---- Step 2: Network Config ----
st.header("2. Network Configuration")
n_layers = st.number_input("Hidden layers", 1, 3, st.session_state["network_config"]["layers"])
n_neurons = st.number_input("Neurons per hidden layer", 4, 128, st.session_state["network_config"]["neurons"])
st.session_state["network_config"] = {"layers": n_layers, "neurons": n_neurons}

# ---- Step 3: Feature Extraction ----
st.header("3. Feature Extraction & Training")
if len(st.session_state["sounds"]) >= 2 and len(set(st.session_state["labels"])) == 2:
    X = []
    y = []
    label_to_int = {name: i for i, name in enumerate(st.session_state["label_names"])}
    for snd, lbl in zip(st.session_state["sounds"], st.session_state["labels"]):
        feats = extract_features(snd)
        X.append(feats)
        y += [label_to_int[lbl]] * feats.shape[0]
    X = np.vstack(X)
    y = np.array(y)
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1

    # ---- Step 4: Train Model ----
    if st.button("Train Neural Network"):
        model = build_model(n_layers, n_neurons)
        history = model.fit(X, y_onehot, epochs=10, batch_size=32, verbose=0)
        st.session_state["model"] = model
        st.session_state["feature_data"] = X
        st.session_state["feature_labels"] = y
        st.success("Neural Network trained!")

        # Show accuracy
        preds = np.argmax(model.predict(X), axis=1)
        acc = np.mean(preds == y)
        st.write(f"Training accuracy: {acc:.2%}")

        # ---- Step 5: Visualize NN ----
        st.header("5. Neural Network Visualization")
        fig = visualize_network(model)
        st.pyplot(fig)
else:
    st.warning("Please record at least 2 sounds with different labels.")

# ---- Step 6: Test New Sound ----
st.header("6. Test the Model")
if st.session_state["model"] is not None:
    test_audio = st.audio_input("Record/upload a new sound to classify", key="test_audio")
    if test_audio is not None:
        feats = extract_features(test_audio)
        preds = st.session_state["model"].predict(feats)
        avg_pred = np.mean(preds, axis=0)
        pred_label = st.session_state["label_names"][np.argmax(avg_pred)]
        st.write(f"Predicted label: **{pred_label}** (confidence: {100 * np.max(avg_pred):.1f}%)")

# ---- FAQ / TIPS ----
with st.expander("ℹ️ How this works"):
    st.write(
        """
        - Record or upload two different types of sounds and label them.
        - Each sound is split into 100 chunks, and a frequency analysis (FFT) is used to extract 16 features per chunk.
        - A neural network is trained to distinguish between the two sound types.
        - You can change the number of hidden layers and neurons.
        - Once trained, record a new sound to see the model guess the label
        """
    )
