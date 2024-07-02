import pandas as pd
import numpy as np

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