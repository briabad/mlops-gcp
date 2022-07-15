
import tensorflow as tf
import tensorflow_transform as tft
import pandas as pd 


import autoencoder_constant

NUMERICAL_FEATURES = autoencoder_constant.NUMERICAL_FEATURES
def t_name(key):
    """
    Rename the feature keys so that they don't clash with the raw keys when
    running the Evaluator component.
    Args:
    key: The original feature key
    Returns:
    key with '_xf' appended
    """
    return key + '_xf'

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
    inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in NUMERICAL_FEATURES:
        # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
        outputs[key] = tft.scale_to_0_1(inputs[key])

    return outputs

