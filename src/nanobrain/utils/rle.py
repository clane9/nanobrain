"""Run-length encoding utils

Copied from:
https://github.com/mikedh/trimesh/blob/5.1.0/trimesh/voxel/runlength.py
"""

import numpy as np


def split_long_brle_lengths(lengths, dtype=np.int64):
    """
    Split lengths that exceed max dtype value.

    Lengths `l` are converted into [max_val, 0] * l // max_val + [l % max_val]

    e.g. for dtype=np.uint8 (max_value == 255)
    ```
    split_long_brle_lengths([600, 300, 2, 6], np.uint8) == \
             [255, 0, 255, 0, 90, 255, 0, 45, 2, 6]
    ```
    """
    lengths = np.asarray(lengths)
    max_val = np.iinfo(dtype).max
    bad_length_mask = lengths > max_val
    if np.any(bad_length_mask):
        # there are some bad lengths
        nl = len(lengths)
        repeats = np.asarray(lengths) // max_val
        remainders = (lengths % max_val).astype(dtype)

        lengths = np.concatenate(
            [
                np.array([max_val, 0] * repeat + [remainder], dtype=dtype)
                for repeat, remainder in zip(repeats, remainders)
            ]
        )
        lengths = lengths.reshape((np.sum(repeats) * 2 + nl,)).astype(dtype)
        return lengths
    elif lengths.dtype != dtype:
        return lengths.astype(dtype)
    else:
        return lengths


def dense_to_brle(dense_data, dtype=np.int64):
    """
    Get the binary run length encoding of `dense_data`.

    Parameters
    ----------
    dense_data: rank 1 bool array of data to encode.
    dtype: numpy int type.

    Returns
    ----------
    Binary run length encoded rank 1 array of dtype `dtype`.

    Raises
    ----------
    ValuError if dense_data is not a rank 1 bool array.
    """
    if dense_data.dtype != bool:
        raise ValueError("`dense_data` must be bool")
    if len(dense_data.shape) != 1:
        raise ValueError("`dense_data` must be rank 1.")
    n = len(dense_data)
    starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    lengths = split_long_brle_lengths(lengths, dtype=dtype)
    if dense_data[0]:
        lengths = np.pad(lengths, [1, 0], mode="constant")
    return lengths


_ft = np.array([False, True], dtype=bool)


def brle_to_dense(brle_data, vals=None):
    """Decode binary run length encoded data to dense.

    Parameters
    ----------
    brle_data: BRLE counts of False/True values
    vals: if not `None`, a length 2 array/list/tuple with False/True substitute
        values, e.g. brle_to_dense([2, 3, 1, 0], [7, 9]) == [7, 7, 9, 9, 9, 7]

    Returns
    ----------
    rank 1 dense data of dtype `bool if vals is None else vals.dtype`

    Raises
    ----------
    ValueError if vals it not None and shape is not (2,)
    """
    if vals is None:
        vals = _ft
    else:
        vals = np.asarray(vals)
        if vals.shape != (2,):
            raise ValueError(f"vals.shape must be (2,), got {vals.shape}")
    ft = np.repeat(_ft[np.newaxis, :], (len(brle_data) + 1) // 2, axis=0).flatten()
    return np.repeat(ft[: len(brle_data)], np.asarray(brle_data, dtype=int)).flatten()
