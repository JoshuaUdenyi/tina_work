"""Engineering checks only. Random weights here are never a trained-model result."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import cv2
from visual_demo.core import preprocess_bytes,variants,load_checkpoint,predict,gradcam,feature_maps,validate_probabilities
from visual_demo.results import GRID,load_results,audit_results
from src.models.proposed import build_proposed

class DemoTests(unittest.TestCase):
    def test_archived_result_arithmetic(self):
        self.assertEqual(len(audit_results(load_results())),4)

    def test_preprocessing_rgb_and_area(self):
        bgr=np.zeros((83,91,3),np.uint8);bgr[:,:,2]=255;bgr[20:50,:,1]=83
        ok,enc=cv2.imencode('.png',bgr);self.assertTrue(ok)
        actual=preprocess_bytes(enc.tobytes())
        expected=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        np.testing.assert_array_equal(actual,expected)
        with self.assertRaises(ValueError):preprocess_bytes(b'not an image')

    def test_degradation_reproducibility_and_rng_restoration(self):
        image=np.linspace(0,1,64*64*3,dtype=np.float32).reshape(64,64,3)
        np.random.seed(391);state=np.random.get_state();expected=np.random.random(8);np.random.set_state(state)
        for kind in GRID:
            a=variants(image,kind,42);b=variants(image,kind,42)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[0],image)
            self.assertEqual(a.shape,(4,64,64,3));self.assertTrue(np.isfinite(a).all());self.assertTrue(((a>=0)&(a<=1)).all())
        np.testing.assert_array_equal(np.random.random(8),expected)
        lights=variants(np.ones((64,64,3),np.float32)*.5,'low_light',42)
        self.assertGreater(lights[1].mean(),lights[2].mean());self.assertGreater(lights[2].mean(),lights[3].mean())

    def test_checkpoint_roundtrip_and_explanations(self):
        m=build_proposed();self.assertEqual(m.count_params(),61323)
        x=np.random.default_rng(42).random((2,64,64,3),dtype=np.float32)
        expected=predict(m,x)
        with tempfile.TemporaryDirectory() as tmp:
            full=Path(tmp)/'engineering_only.keras';weights=Path(tmp)/'engineering_only.weights.h5'
            m.save(full);m.save_weights(weights)
            for path,mode in [(full,'Full saved model'),(weights,'Proposed weights only')]:
                loaded=load_checkpoint(path,mode);np.testing.assert_allclose(predict(loaded,x),expected,atol=1e-7)
                hm=gradcam(loaded,x[0],int(expected[0].argmax()))
                self.assertEqual(hm.shape,(4,4));self.assertTrue(np.isfinite(hm).all())
                np.testing.assert_allclose(predict(loaded,x),expected,atol=1e-7)
                self.assertEqual(feature_maps(loaded,x[0],'conv_init_relu6').shape,(32,32,32))
        with self.assertRaises(FileNotFoundError):load_checkpoint('/no/such/weights.keras')
        with self.assertRaises(ValueError):validate_probabilities(np.ones((2,43)),2)

if __name__=='__main__':unittest.main(verbosity=2)
