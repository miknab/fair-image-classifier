# Standard library imports
import os
from typing import List

# 3rd party imports
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import random


def set_seed(seed: int = 42) -> int:
    """
    Sets all relevant seeds and returns the seed value.

    Sets seeds by defining environment variables,
    tensorflow, numpy and random seeds. The value
    of the seed is returned so it can be used in 
    other places if required ensuring consistency.

    Keyword arguments:
    seed -- value of the seed to be used (default: 42)

    Returns:
    seed -- value of the seed to be used (default: 42)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def pxlstring2pxlvec(df: pd.DataFrame, 
                     idx: int, 
                     colname: str = "pixels",
                     sep: str = " "
                    ) -> np.ndarray:
    """
    Converts a string of pixel values to a vector of pixel values.
    
    The pixel string at row index "idx" in the "colname" column of
    "df" is converted into a 1D array of integers. This corresponds
    to a vectorized array of pixel values.

    Arguments:
    df  -- dataframe containing the pixel value strings
    idx -- row index of df to be selected

    Keyword arguments:
    colname -- name of column in df containing the pixel value strings
               (default: "pixels")
    sep     -- separator used between pixel values in pixel string 
               (default: " ")
    
    Returns:
    img -- vector of pixel values
    """
    # Check if colname is a valid column name
    if colname not in df.columns:
        raise ValueError("colname must be an existing column name of "
                         + "the pd.DataFrame df."
                        )

    # Extract the pixel string at row index idx (that's the only
    # pixel string considered in this execution of the pxlstring2pxlvec
    # function).
    pxl_str = df[colname][idx]
    pxl_list = pxl_str.split(sep)
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