# Standard library imports
import os
from typing import List

# 3rd party imports
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import random


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def pxlstring2pxlvec(df: pd.DataFrame, idx: int) -> np.ndarray:
    """
    Takes in a string of space-separated integers and converts it into
    an 1D array of integers corresponding to the vectorized array of
    pixel values.
    """
    pxl_str = df["pixels"][idx]
    pxl_list = pxl_str.split(" ")
    n_pxls = len(pxl_list)

    img_dim = np.sqrt(n_pxls)
    assert float(int(img_dim)) == img_dim
    img_dim = int(img_dim)

    img = np.array(pxl_list, dtype=int)
    return img
    
def pxlvec2pxlarray(pxlvec: np.ndarray) -> np.ndarray:
    """
    Takes in a pixel vector and converts it into an array of integers
    corresponding to the array of pixel values.
    """
    vec_dim = pxlvec.shape[0]
    arr_dim = int(np.sqrt(vec_dim))
    return pxlvec.reshape(arr_dim, arr_dim)

def upsample_image(image: np.array, new_dim: int) -> np.array:
    pil_image = Image.fromarray(image.astype(np.uint8))  # Convert numpy array to PIL Image
    upsampled_image = pil_image.resize((new_dim, new_dim), Image.LANCZOS)
    return np.array(upsampled_image)  # Convert PIL Image back to numpy array and normalize

def preproc_data(X: np.array, add_channels_dim = True) -> np.array:
    if add_channels_dim:
        X = X.reshape(X.shape + (1,))
    X = X.astype("float32")
    return X / 255