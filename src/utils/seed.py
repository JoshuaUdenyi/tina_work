import os
import random
import numpy as np

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except ImportError:
        pass
