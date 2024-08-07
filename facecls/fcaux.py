# Standard library imports
import os
from typing import List
import math

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

    Parameters
    ----------
    seed : int
        value of the seed to be used
        default: 42

    Returns
    -------
    int
        value of the seed to be used
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

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing the pixel value strings
    idx : int
        row index of df to be selected
    colname : str
        name of column in df containing the pixel value strings
        default: "pixels"
    sep : str
        separator used between pixel values in pixel string 
        default: " "
    
    Returns
    -------
    numpy.ndarray
        vector of pixel values

    Raises
    ------
    ValueError
        If "colname" is not a valid column name of "df".
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
    Converts pixel vector into 2D pixel array.
    
    Takes in a pixel vector and converts it into an array of integers
    corresponding to the array of pixel values.

    Notes
    -----
    The pxlvec vector is assumed to have length equal to a square
    number such that it can be reshaped into a square array.
    
    Parameters
    ----------
    pxlvec : numpy.ndarray
        1D array containing pixel vectors

    Returns
    -------
    numpy.ndarray
        2D array containing the pixel vectors

    Raises
    ------
    ValueError
        If pxlvec is not a column or a row vector
    AssertionError
        If pxlvec still has wrong dimensions after reshaping
    ValueError
        If length of pxlvec is not a square number (see Notes section)
    """
    # Check that pxlvec is indeed a column or a row vector
    if len(pxlvec.shape) > 2 or (len(pxlvec.shape) == 2 and 1 not in pxlvec.shape):
        raise ValueError("pxlvec must be a column vector of dimension (n,1) "
                         + "or a row vector of dimension (1,n) or (n,), respectively."
                        )
        
    # If pxlvec is a column vector of dimension (n,1) or a row vector
    # of dimesnion (1,n), convert it into a proper 1D array of dimension
    # (n,).
    if len(pxlvec.shape) == 2:
        if pxlvec.shape[1]==1:
            pxlvec = pxlvec.transpose()[0]
        else:
            pxlvec = pxlvec[0]

    # Just make really sure the dimensions are correct
    assert len(pxlvec.shape)==1, "pxlvec still has the wrong dimensions --> ABORTING."

    # Get the length of pxlvec
    vec_dim = pxlvec.shape[0]

    # ... and use that to compute the dimensions of the 
    # resulting image. We use math.isqrt instead of numpy.sqrt
    # in order to get the rounded-down integer in case 
    # the actual square root is a decimal number. This will
    # be used below to check if the pxlvec vector can be
    # reshaped into a square 2D array or not.
    side = math.isqrt(vec_dim)

    # Check if the pixel vector can be reshaped into a
    # square pixel array
    if side ** 2 != vec_dim:
        raise ValueError("Length of pxlvec has no integer square root, i.e. it cannot be "
                         + "reshaped into a square 2D array."
                        )
    return pxlvec.reshape(side, side)

def upsample_image(image: np.ndarray, new_dim: int) -> np.ndarray:
    """
    Upsamples the image to dimensions new_dim x new_dim.

    The image is passed in as a numpy.ndarray, converted into a PIL image,
    upsampled to the new dimensions (using a Lanczos filter), converted back
    into a numpy.ndarray and returned.

    Notes
    -----
    The Lanczos filter is comparatively expensive, which is fine as long as
    the images processed are not too big. This code was written with images
    of size 48x48 pixels in mind.
    
    Parameters
    ----------
    image : np.ndarray
        Array containing the pixel values of an image.
        
    Returns
    -------
    np.ndarray
        Array containing the pixel values of the original but upsampled image.
    """
    # Convert numpy array to PIL Image so we can use the pil_image.resize function
    pil_image = Image.fromarray(image.astype(np.uint8))
    upsampled_image = pil_image.resize((new_dim, new_dim), Image.LANCZOS)

    # Convert PIL Image back to numpy array and return
    return np.array(upsampled_image)  

def preproc_data(X: np.ndarray, add_channels_dim = True) -> np.ndarray:
    """
    Normalize the pixel values, convert them to floats and add a 
    channel (optional).

    The pixel values in X are normalized from [0,255] to [0,1],
    converted to floats of type "float32" and optionally one can 
    add a new channel (e.g. a color channel) if desired.

    Notes
    -----
    - This function assumes that the image X originally contains values
      between 0 and 255.
    - The channel has to be added if the images will be used in any
      form of CNN architecture. Only for processing by multilayer
      perceptrons (MLP) there is no need for an additional channel.

    Parameters
    ----------
    X : np.ndarray
        The pixel array of the image to be preprocessed
    add_channels_dim : bool
        Truth value defining whether or not a channel shall be added
        Default: True

    Returns
    -------
    np.ndarray
        Array of the image with normalized pixel values and optional,
        additional channel

    Raises
    ------
    ValueError
        If largest pixel value in the original X is either <= 1 or >255.
    """
    # Check if there is evidence for unexpected pixel value ranges
    if X.max() <= 1 or X.max() > 255:
        raise ValueError(f"The maximal pixel value of X = {X.max()} and "
                         + "thus indicates that the pixel values are not "
                         + "defined over the interval [0,255] as expected."
                        )

    # If requested, add a channel to the image. 
    if add_channels_dim:
        X = X.reshape(X.shape + (1,))

    # Convert pixel values to float32
    X = X.astype("float32")

    # Normalize and return
    return X / 255