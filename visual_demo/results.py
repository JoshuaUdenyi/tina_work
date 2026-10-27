"""Saved-result readers. No TensorFlow and no inferred predictions."""
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RUNS = ['proposed_aware', 'proposed_clean_only', 'mobilenetv2_aware', 'squeezenet_aware']
LABELS = ['Proposed aware', 'Proposed clean-only', 'MobileNetV2 aware', 'SqueezeNet aware']
GRID = {
    'fog': ['light', 'moderate', 'heavy'],
    'motion_blur': [5, 9, 15],
    'low_light': [0.7, 0.5, 0.3],
    'gaussian_noise': [0.02, 0.05, 0.1],
    'rain': ['light', 'moderate', 'heavy'],
    'occlusion': [0.1, 0.25, 0.4],
}
TITLES = {'fog':'Fog', 'motion_blur':'Motion blur', 'low_light':'Low light',
          'gaussian_noise':'Gaussian noise', 'rain':'Rain', 'occlusion':'Occlusion'}

def load_results():
    return {r: json.loads((ROOT/'results'/r/'eval.json').read_text()) for r in RUNS}

def audit_results(data):
    """Check internal consistency, not experiment provenance or split validity."""
    notes = []
    for run, item in data.items():
        e = item['evaluation']; cm = np.asarray(e['clean']['confusion_matrix'])
        assert cm.shape == (43,43), f'{run}: expected 43 by 43 confusion matrix'
        assert cm.sum() == item['num_test'], f'{run}: test count mismatch'
        assert np.isclose(np.trace(cm)/cm.sum(), e['clean']['accuracy']), f'{run}: accuracy mismatch'
        vals = [m['accuracy'] for m in e['per_condition'].values()]
        assert len(vals) == 18, f'{run}: expected 18 conditions'
        assert np.isclose(np.mean(vals), e['mean_degraded_accuracy']), f'{run}: mean mismatch'
        assert np.isclose(np.mean(vals)/e['clean']['accuracy'], e['robustness_score']), f'{run}: R mismatch'
        notes.append(f'{run}: counts, clean accuracy, 18-condition mean and R are consistent.')
    return notes

def rows(data):
    return [dict(Model=LABELS[i], Clean=100*data[r]['evaluation']['clean']['accuracy'],
                 Degraded=100*data[r]['evaluation']['mean_degraded_accuracy'],
                 R=data[r]['evaluation']['robustness_score'], Parameters=data[r]['params'],
                 Samples=data[r]['num_test']) for i,r in enumerate(RUNS)]
