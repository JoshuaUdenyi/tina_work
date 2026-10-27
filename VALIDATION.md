# Validation performed

Package prepared 27 October 2026 from `Tina (3).zip`.

## Passed

- Inspected both outer ZIP and its nested Colab ZIP: no trained model checkpoints or sign photographs found.
- Built the supplied TensorFlow/Keras proposed architecture: 61,323 parameters; input 64×64×3; output 43; final activation 4×4×256.
- All four archived main runs: clean confusion counts total 7,950 and reproduce reported clean accuracy. All 18-condition means and R ratios agree with saved values. Smoke-test results excluded.
- Four unittest cases passed: archived arithmetic; OpenCV RGB/INTER_AREA/normalization equivalence and invalid image handling; repeatability/range checks for all six degradations and restoration of NumPy RNG state; complete-model and weights-only loading equivalence plus Grad-CAM and feature maps.
- Streamlit AppTest: initial app renders without exceptions, four metric values match saved results, degradation selector responds, inference button remains disabled when no checkpoint exists.
- Streamlit AppTest with an injected completed engineering run: comparison result rendering, Grad-CAM generation and feature-map controls complete without app errors.
- Offline viewer JavaScript parsed and executed in a minimal DOM harness: default two-series display; four-series switch; severity setting update; four comparison rows; CSV download action.
- All five generated thesis PNG figures visually inspected; heatmap cell sizing corrected after inspection. Vector PDFs exported from the same figure objects.

## Test environment

Python 3.12; TensorFlow CPU 2.21.0; Keras 3.15.1; Streamlit 1.66.1; OpenCV 5.0.0; NumPy 2.5.4. Dependency version ranges are provided, but all combinations within those ranges have not been tested. Use your original TensorFlow environment if an old checkpoint has version-specific serialization requirements.

## Not validated

Your actual trained weights, image classification performance on new examples, original test split provenance, or class names could not be validated because those files were absent. Random-weight checkpoints were created only in temporary test directories and deleted; they are not bundled and their scores are not thesis evidence.

A full browser screenshot test of OPEN_RESULTS.html was attempted, but the Chromium download was unavailable/corrupt in this environment. The JavaScript interaction harness passed; desktop/mobile browser appearance was not independently screenshot-verified. Streamlit was tested with its AppTest framework, not through a browser video session.
