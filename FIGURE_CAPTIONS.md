# Suggested thesis figure captions and explanations

Choose figure numbers based on where these are inserted. Figures have no fabricated run results. Cite the project source and archived experiment outputs as their origin, rather than attributing them to external literature.

## 01 Architecture

**Caption:** Structure of the proposed lightweight traffic-sign classifier, showing preprocessing, depthwise separable blocks, tensor dimensions and the classification head. Source: project source code, `src/models/proposed.py`.

**Explanation:** A normalized 64×64 RGB input passes through an initial stride-2 convolution and four depthwise separable blocks. Spatial dimensions decrease to 4×4 while channel depth increases to 256. Global average pooling converts this tensor to a feature vector, followed by dropout and a 43-output softmax layer. The default implementation contains 61,323 total parameters. This figure describes the implemented architecture; it does not itself show that a trained model achieves any particular accuracy.

## 02 Model comparison

**Caption:** Clean-input accuracy and mean accuracy across eighteen simulated degradation conditions for the four archived model configurations. Source: supplied per-run evaluation JSON files; each reports 7,950 test images.

**Explanation:** Degradation-aware training increases mean degraded accuracy of the proposed architecture from 55.77% to 73.98%, an improvement of 18.21 percentage points. Clean accuracy changes from 93.12% to 91.25%, a decrease of 1.87 percentage points. The appropriate conclusion is an improved degradation trade-off for the same proposed architecture. MobileNetV2 achieves higher archived clean and degraded accuracy, with a much larger parameter count, so the proposed model should not be described as the most accurate model overall.

## 03 Robustness curves

**Caption:** Proposed-aware and proposed-clean-only accuracy over three severity settings for each of six simulated degradation types. Source: archived per-condition evaluation metrics.

**Explanation:** Each panel isolates one degradation type. The horizontal positions correspond to increasing severity, with actual source settings printed beneath the axis. These are eighteen discrete conditions, not a continuous estimate of all possible environmental conditions. The comparison makes degradation-specific weaknesses and the training-strategy trade-off visible. Low-light gamma decreases as severity increases under the implementation’s inverse-gamma convention. Lines are visual guides, not additional observations.

## 04 Condition heatmap

**Caption:** Accuracy of the proposed degradation-aware model for six degradation types at low, medium and high severity. Source: `results/proposed_aware/eval.json`.

**Explanation:** Each cell reports accuracy from the saved evaluation and displays its actual severity setting. The heavy-occlusion setting has approximately 33.35% accuracy, showing an important limitation even after degradation-aware training. Colours share a fixed 0–100% scale to avoid overstating differences. The map summarizes saved dataset measurements and does not display predictions for a newly uploaded image.

## 05 Clean confusion matrix

**Caption:** Row-normalized confusion matrix for the proposed degradation-aware model on clean test images. Source: archived clean-input confusion counts in `results/proposed_aware/eval.json`.

**Explanation:** Rows represent true class IDs and columns represent predicted IDs. Values are normalized within each true class so performance on less frequent classes remains visible. A strong diagonal indicates correct predictions, while off-diagonal intensity shows confusions. Tick labels use numeric IDs because the uploaded source did not include a verified mapping to class names. Overall accuracy is obtained from the raw diagonal sum divided by the total count, not the unweighted mean of the normalized diagonal.

## Optional live figures

**Caption template:** Predicted classes and softmax scores for one held-out traffic-sign image under [degradation], using the supplied trained [checkpoint name]. The corruption seed is [seed], and ground truth is [class ID, if known].

**Explanation:** Describe only the observed predictions, including failures or confidence drops. A single-image demonstration is qualitative evidence of model behaviour and does not estimate dataset accuracy. Record the checkpoint hash and input hash from the export manifest.

**Grad-CAM caption template:** Grad-CAM overlays for class [ID], computed from the final proposed activation `ds_pw_relu6_3`. Heatmaps originate at 4×4 spatial resolution and are enlarged for display.

**Explanation:** Brighter regions indicate stronger positive contribution to the selected class score under the visualization method. The overlay is coarse and does not establish causal understanding, localization accuracy or robustness by itself.
