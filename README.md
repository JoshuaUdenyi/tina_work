# Traffic sign robustness visual demonstration

This package turns the supplied TINA source and archived evaluation results into a supervisor demonstration. It includes an offline results explorer, a TensorFlow/Keras Streamlit app, five thesis figures in PNG and vector PDF, and code to export real prediction comparisons once a checkpoint is supplied.

## Start here

**To show the saved results immediately:** double-click `OPEN_RESULTS.html`. It works offline without installation. Change the degradation dropdown and compare the proposed aware/clean-only models or all four models. This page displays archived metrics; it never performs inference.

**To show a sign being classified:** follow the setup below and add your real trained checkpoint. The supplied `Tina (3).zip`, including `colab/tINA_colab.zip`, contained no `.keras`, `.h5` or TensorFlow checkpoint files and no original traffic-sign photographs. Saved metrics cannot be converted back into trained weights. No fake predictions or pre-trained substitute models are included.

## 1. Install on Windows

Extract the ZIP, open PowerShell in the extracted `traffic_sign_visual_kit` folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Using the venv Python directly avoids PowerShell activation-policy problems. Python 3.12 is the recommended environment. The first installation needs internet access; model inference and the saved-results dashboard run locally afterward. CPU inference is sufficient for the demonstration.

On macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python -m streamlit run app.py
```

## 2. Add the trained model

Copy your degradation-aware proposed model here:

```text
checkpoints/proposed_aware/best.keras
```

Optionally copy the clean-only checkpoint here:

```text
checkpoints/proposed_clean_only/best.keras
```

Enter its path in the optional comparison field in the app. The comparison receives the identical clean/degraded pixels as the main model. Both models use the selected loading mode.

- A complete model saved by `model.save(...)`: choose **Full saved model**. Supported by Keras loading for `.keras` and compatible `.h5` full models.
- Proposed architecture weights saved by `model.save_weights(...)`: choose **Proposed weights only** and enter the actual file path, usually ending `.weights.h5`. The exact `build_proposed` architecture is constructed before strict weight loading. There is no `skip_mismatch` fallback.
- A TensorFlow SavedModel directory or TensorFlow checkpoint prefix is not handled by this loader. Reopen it in the original training environment and export a complete `.keras` model, or supply the original environment details to adapt loading.
- If loading fails due to a Keras version difference, use your original training environment to load and re-save the model. Keep the original file.

The uploaded evaluation metadata records `/content/checkpoints/proposed_aware/best.keras`, so the checkpoint may still be in the Colab runtime, Google Drive, or your original results download. If the runtime still contains it, run this **in Colab**:

```python
from google.colab import files
from pathlib import Path
p = Path('/content/checkpoints/proposed_aware/best.keras')
if not p.is_file():
    raise FileNotFoundError('Find the original trained best.keras first; do not retrain or substitute silently.')
files.download(str(p))
```

Use the checkpoint that produced the submitted thesis results. A newly trained checkpoint is a new experiment and may give different results.

## 3. Demonstrate the model

1. Open **Live image demonstration** and load the checkpoint path.
2. Upload a PNG/JPG/PPM image tightly cropped around one traffic sign. Use a held-out test image where possible. The classifier does not detect/crop signs automatically.
3. If you know its actual class, select the true class ID. This enables correct/incorrect labels. Otherwise correctness remains unknown.
4. Choose fog, motion blur, low light, Gaussian noise, rain, or occlusion. Keep seed 42 to reproduce the same illustration.
5. Click **Run visual demonstration**. Inspect clean, low, medium and high severity images and actual predictions.
6. Use the severity slider to inspect the top five outputs and optional explanation for one degraded input.
7. Load the clean-only checkpoint to show the effect of training strategy on the same pixels. Include both successes and failures.
8. Download the 300 dpi prediction comparison and JSON containing scores, corruption seed and checkpoint SHA-256 identifiers.
9. Under **Grad-CAM and feature maps**, generate the coarse class-sensitive overlay and the first eight initial-layer feature maps. These tools target the supplied proposed architecture, not arbitrary baseline heads.

The archive supplies numeric class IDs, not a named class map. The app therefore defaults to `Class 0` through `Class 42`. To show names such as “Stop”, supply a JSON list of 43 names in your verified output order, or an object with keys `"0"` through `"42"`. Do not reorder model outputs or guess a label map from alphabetical names. Example structure only:

```python
# Replace each value with your verified dataset label name.
import json
names = [f'Class {i}' for i in range(43)]
with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(names, f, indent=2)
```

## 4. Thesis figures

Ready to use in `thesis_figures/`:

| File stem | What it shows | Evidence |
|---|---|---|
| `01_architecture` | Exact proposed architecture, tensor sizes and explanation layer | Supplied `src/models/proposed.py`; parameter count checked by building it |
| `02_model_comparison` | Clean vs mean degraded accuracy for four archived runs | Supplied per-run `eval.json` |
| `03_robustness_curves` | Aware vs clean-only accuracy at all 18 degradation settings | Supplied per-condition metrics |
| `04_condition_heatmap` | Proposed-aware accuracy by degradation and severity | Supplied per-condition metrics |
| `05_confusion_clean` | Row-normalized clean-input confusion matrix | Supplied proposed-aware confusion counts |

Every figure has a 300 dpi PNG and a vector PDF. Read `FIGURE_CAPTIONS.md` before inserting them; assign figure numbers to fit your thesis. The exporter does not invent thesis numbering. `figure_manifest.json` records source file hashes and the arithmetic checks.

Regenerate archived figures:

```powershell
.\.venv\Scripts\python.exe export_figures.py
```

Generate **real single-image prediction figures** for all six degradations:

```powershell
.\.venv\Scripts\python.exe export_figures.py --image "C:\path\to\held_out_sign.png" --checkpoint "checkpoints\proposed_aware\best.keras" --true-class 14 --seed 42
```

Class 14 here is only an example; set the known label of your selected image. Omit `--true-class` if unknown. Add `--weights-only` for a proposed weights-only HDF5 file, `--class-names class_names.json` for your verified mapping, or `--out another_folder` to preserve earlier exports. Prediction files start `live_`; they are only created when an actual checkpoint and input image are supplied.

## 5. Implementation details matched to the source

- Input: `(None, 64, 64, 3)`, RGB `float32`, scaled to `[0,1]`.
- Preprocessing: OpenCV decode, BGR→RGB, `INTER_AREA` resize to 64×64, then divide by 255. The preprocessing implementation was located in the nested `colab/tINA_colab.zip`; the outer ZIP omitted `src/data/`.
- Proposed model: initial 3×3 convolution with 32 filters and stride 2; four depthwise/pointwise blocks with output channels `[32,64,128,256]`, strides `[1,2,2,2]`; BN/ReLU6 after each convolution; global average pooling; dropout 0.30; Dense(43, softmax).
- Total parameters: **61,323**, including BN trainable/nontrainable parameters. This matches the default source and archived metadata.
- Last proposed activation: **`ds_pw_relu6_3`**, `(None,4,4,256)`. The final pointwise convolution is `ds_pw_3`; Grad-CAM uses the following activation.
- Low light: `output = brightness × input^(1/gamma)`, gamma `[0.7,0.5,0.3]`, brightness `[0.90,0.75,0.60]`. The inverse exponent is essential.
- Motion-blur kernels `[5,9,15]`; Gaussian sigma `[0.02,0.05,0.10]`; occlusion requested fractions `[0.10,0.25,0.40]`; fog and rain `light/moderate/heavy`.
- `src/augment/degradations.py` is copied unchanged. Rain is exactly the uploaded streak implementation; its comments do not make it physically realistic rain. Occlusion fractions are targets before integer rounding/capping.
- Degradation is applied **after resizing**, as in the evaluation code. Reapplying it to a full-resolution upload would change its meaning.
- The original corruption functions use a global NumPy RNG. The wrapper locks, seeds and restores that RNG. Levels use the same seed to produce controlled visual illustrations. This does not reproduce the original evaluation’s global draw order.
- Full models load with `compile=False, safe_mode=True`, and input/output shape and softmax values are checked. No fallback creates an untrained model for inference.
- Grad-CAM computes gradients of a **pre-softmax score**, reconstructing the supplied global-pool/Dense head. It does not mutate the cached model's activation. Grad-CAM is an interpretive visualization, not proof that the model reasons correctly. Its 4×4 spatial support limits detail.

Minimal use of the actual implementation:

```python
from pathlib import Path
from visual_demo.core import load_checkpoint, preprocess_bytes, variants, predict

model = load_checkpoint('checkpoints/proposed_aware/best.keras')
image = preprocess_bytes(Path('held_out_sign.png').read_bytes())
images = variants(image, 'fog', seed=42)  # clean + light + moderate + heavy
probabilities = predict(model, images)   # (4, 43), genuine model outputs
print(probabilities.argmax(axis=1))
```

## 6. What the results do and do not establish

The supplied archived evaluation reports 7,950 test images for each main run. Proposed aware has 91.2453% clean accuracy, 73.9818% mean degraded accuracy and R=0.810802. Proposed clean-only has 93.1195%, 55.7687% and R=0.598894, respectively. Thus mean degraded accuracy improves by **18.21 percentage points**, at a **1.87 percentage-point clean accuracy cost**. These are saved experiment values, not measurements generated by this package.

The tools checked that confusion-matrix totals equal the reported sample counts, clean accuracy equals trace/total, the 18-condition mean matches its reported value, and R matches degraded/clean. This establishes internal arithmetic consistency, not independent verification of training, dataset provenance, class mapping or track-disjoint splits. Original images, checkpoints and split manifests are needed for that.

A few image examples do not establish test-set accuracy. Softmax confidence is not calibrated correctness, nor the thesis robustness score. Do not report the four severities of one image as dataset accuracy. Synthetic corruptions are controlled tests, not field testing. `proposed_smoke` was excluded from all thesis plots.

## 7. Supervisor presentation sequence

- **30 seconds:** explain that the model classifies one cropped sign among 43 classes.
- **1 minute:** show `01_architecture` and explain the compact depthwise/pointwise blocks.
- **2 minutes:** run one correctly labeled held-out sign, progressively degrade it, and compare both checkpoints.
- **1 minute:** show a failure example and the limits of the coarse Grad-CAM overlay.
- **1 minute:** show the archived ablation graph and per-condition curves. Explain the +18.21 pp degraded gain and clean accuracy cost.
- **30 seconds:** distinguish illustrative single-image inference from the archived full test-set evaluation.

Prepare several genuine held-out images, including a difficult case, before the meeting. The default app takes camera snapshots as an alternative input, but those are still static-image classification examples, not video tracking or detection.

## 8. Code checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary random-weight checkpoints only to verify model-loading equivalence, shapes, preprocessing, RNG reproducibility, feature maps and Grad-CAM. Those checkpoints are deleted and never included as trained artifacts. They do not validate your model’s recognition accuracy. See `VALIDATION.md` for the actual checks run while building this package.

## Technical documentation

- Keras Grad-CAM example: https://keras.io/examples/vision/grad_cam/
- TensorFlow saved-model loading: https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
- Streamlit file upload: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit file download: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button

This is a demonstration add-on, not a replacement training repository. Original supplied architecture and degradation code are retained under `src/`; existing archived run results are preserved under `results/`. No retraining was performed.
