"""Inference uses the uploaded architecture and degradation source unchanged."""
from pathlib import Path
import hashlib
import json
import threading
import cv2
import numpy as np
from visual_demo.results import GRID, ROOT
from src.augment.degradations import apply_degradation_by_name

# The source degradation module uses NumPy's legacy global RNG.
# Serialize calls and restore its state so Streamlit sessions do not interfere.
RNG_LOCK = threading.RLock()

def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def preprocess_bytes(raw):
    # Identical decode/color/resize/normalization steps to nested src/data/preprocess.py.
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Cannot decode this image. Use JPG, PNG or PPM.')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (64,64), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0

def variants(image, kind, seed=42):
    if kind not in GRID: raise ValueError('Unknown degradation')
    with RNG_LOCK:
        state = np.random.get_state()
        try:
            out = [image.copy()]
            for severity in GRID[kind]:
                # Same base random draw per level: useful for repeatable visual comparisons.
                # This is a controlled illustration, not a rerun of the archived evaluation.
                np.random.seed(int(seed))
                out.append(apply_degradation_by_name(image.copy(), kind, severity).astype(np.float32))
            return np.stack(out)
        finally: np.random.set_state(state)

def load_checkpoint(path, mode='Full saved model'):
    import tensorflow as tf
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f'Checkpoint missing: {path}')
    if mode == 'Proposed weights only':
        from src.models.proposed import build_proposed
        model = build_proposed(input_shape=(64,64,3), num_classes=43,
                               filters=[32,64,128,256], dropout=0.30)
        model.load_weights(str(path))  # strict matching; no skip_mismatch
    else:
        model = tf.keras.models.load_model(str(path), compile=False, safe_mode=True)
    if tuple(model.input_shape[1:]) != (64,64,3) or tuple(model.output_shape[1:]) != (43,):
        raise ValueError(f'Expected (None,64,64,3) -> (None,43); got {model.input_shape} -> {model.output_shape}')
    probe = np.asarray(model(np.zeros((1,64,64,3),np.float32), training=False))
    validate_probabilities(probe, 1)
    return model

def validate_probabilities(values, n):
    p = np.asarray(values)
    if p.shape != (n,43) or not np.isfinite(p).all(): raise ValueError('Invalid model output shape or nonfinite values')
    if np.any(p < -1e-5) or not np.allclose(p.sum(axis=1), 1, atol=1e-3):
        raise ValueError('Expected 43 softmax probabilities. Model may output logits or require different preprocessing.')
    return p

def predict(model, images):
    return validate_probabilities(model(np.asarray(images,np.float32), training=False), len(images))

def class_names(path=None):
    labels = [
        "Speed Limit (20 km/h)",
        "Speed Limit (30 km/h)",
        "Speed Limit (50 km/h)",
        "Speed Limit (60 km/h)",
        "Speed Limit (70 km/h)",
        "Speed Limit (80 km/h)",
        "End of Speed Limit (80 km/h)",
        "Speed Limit (100 km/h)",
        "Speed Limit (120 km/h)",
        "No Passing",
        "No Passing for Vehicles Over 3.5 Tonnes",
        "Right of Way at Next Intersection",
        "Priority Road",
        "Yield / Give Way",
        "Stop",
        "No Vehicles",
        "Vehicles Over 3.5 Tonnes Prohibited",
        "No Entry",
        "General Caution",
        "Dangerous Curve Left",
        "Dangerous Curve Right",
        "Double Curve",
        "Bumpy Road",
        "Slippery Road",
        "Road Narrows on the Right",
        "Road Work",
        "Traffic Signals",
        "Pedestrians",
        "Children Crossing",
        "Bicycles Crossing",
        "Beware of Ice / Snow",
        "Wild Animals Crossing",
        "End of All Speed and Passing Limits",
        "Turn Right Ahead",
        "Turn Left Ahead",
        "Ahead Only",
        "Straight or Right",
        "Straight or Left",
        "Keep Right",
        "Keep Left",
        "Roundabout Mandatory",
        "End of No Passing",
        "End of No Passing for Vehicles Over 3.5 Tonnes",
    ]

    if path:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = (
            obj
            if isinstance(obj, list)
            else [obj[str(i)] for i in range(43)]
        )

    if len(labels) != 43 or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise ValueError("Expected 43 class names in model output order.")

    return [
        f"Class {i} - {label}"
        for i, label in enumerate(labels)
    ]

def gradcam(model, image, class_id, layer='ds_pw_relu6_3'):
    """Pre-softmax Grad-CAM without mutating the shared model's softmax.

    Uses Dense pred weights explicitly. The supplied proposed model has a 4x4x256
    activation at ds_pw_relu6_3. At inference dropout is inactive. Softmax is
    omitted for gradients only; displayed probabilities still use full model.
    """
    import tensorflow as tf
    head = model.get_layer('pred')
    if not isinstance(head, tf.keras.layers.Dense): raise ValueError('Grad-CAM here requires the proposed Dense pred head.')
    if layer != 'ds_pw_relu6_3': raise ValueError('This visualizer targets the supplied final proposed activation.')
    feature_model = tf.keras.Model(model.inputs, model.get_layer(layer).output)
    with tf.GradientTape() as tape:
        features = feature_model(tf.convert_to_tensor(image[None], tf.float32), training=False)
        tape.watch(features)
        pooled = tf.reduce_mean(features, axis=(1,2))
        logits = tf.linalg.matmul(pooled, head.kernel)
        if head.bias is not None: logits = logits + head.bias
        score = logits[:,int(class_id)]
    grads = tape.gradient(score, features)
    if grads is None: raise ValueError('No gradient path to the selected activation.')
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.nn.relu(tf.reduce_sum(features[0]*weights,axis=-1)).numpy()
    if not np.isfinite(heatmap).all(): raise ValueError('Nonfinite Grad-CAM')
    maximum = float(heatmap.max())
    if maximum > 0: heatmap /= maximum
    return heatmap

def feature_maps(model, image, layer):
    import tensorflow as tf
    view = tf.keras.Model(model.inputs, model.get_layer(layer).output)
    return np.asarray(view(image[None],training=False))[0]
