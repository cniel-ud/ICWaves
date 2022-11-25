"""
Utils functions
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
import numbers

def check_rng(seed):
    """Turn seed into a np.random.Generator instance

    Parameters
    ----------
    seed : None, int or instance of Generator
        If seed is None, return the Generator using the OS entropy.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        seed = np.random.SeedSequence(seed)
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed

    raise ValueError('%r cannot be used to seed a numpy.random.Generator'
                     ' instance' % seed)

def _pick_windows_from_strided(X, n_windows, window_length, rng):
    """
    See https://stackoverflow.com/a/47983009/4292705
    """

    s0, s1 = X.strides
    n_samples, n_features = X.shape
    n_offsets = n_features - window_length + 1

    # Create `n_windows` unique random indexes for each sample
    windows_ids = np.tile(np.arange(n_offsets), (n_samples, 1))
    windows_ids = np.apply_along_axis(rng.choice, 1, windows_ids,
                                      size=n_windows, replace=False)

    # Find all possible `n_offsets` windows for each sample
    windows = as_strided(X,
                         shape=(n_samples, n_offsets, window_length),
                         strides=(s0, s1, s1))

    # Choose `n_windows` per sample
    windows = windows[np.arange(n_samples)[:, None, None],
                      windows_ids[:, :, None],
                      np.arange(window_length)[None, None, :]]
    return windows


def _pick_windows_with_integer_indexing(X, n_windows, window_length, rng):
    """
    See https://stackoverflow.com/a/47982961/4292705
    """

    n_samples, n_features = X.shape
    n_offsets = n_features - window_length + 1

    # Create offsets, the start index of each window
    offsets = np.tile(np.arange(n_offsets), (n_samples, 1))
    chosen_offsets = np.apply_along_axis(rng.choice, 1, offsets,
                                         size=n_windows, replace=False)

    # Build index of columns according to offsets
    col_ids = np.arange(window_length).reshape(1, 1, window_length)
    col_ids = chosen_offsets[:, :, None] + col_ids

    # Pick the windows using advanced indexing
    windows = X[np.arange(n_samples)[:, None, None], col_ids]

    return windows

def pick_random_windows(X, n_windows, window_length, rng=None):
    """
    It picks windows randomly from each row of `X`.

    Parameters
    ----------
    X (numpy.ndarray):
        Matrix with data samples in its rows.
    n_windows (int):
        Number of unique windows per sample. Windows are taken randomly, but
        each window is different.
    window_length (int):
        Lenght of the window
    rng (int, Generator instance, or None):
        Determines random number generation for picking the windows. Use an
        int to make the randomness deterministic.

    Returns
    -------
    Y (numpy.ndarray):
        Windows of rows of `X` selected randomly

    Shapes
    ------
    X : (`n_samples`, `n_features`)
    Y : (`n_samples`, `n_windows`, `window_length`)
    """

    if X.ndim == 1:
        X = X[np.newaxis, :]

    n_samples, n_features = X.shape
    n_bytes_per_element = X.dtype.itemsize
    n_offsets = n_features - window_length + 1
    peak_memory_use = n_samples * n_offsets * window_length \
                      * n_bytes_per_element

    rng = check_rng(rng)

    if peak_memory_use < 1 * 1e9: # 2 GB
        return _pick_windows_from_strided(X, n_windows, window_length,
                                          rng)
    else:
        return _pick_windows_with_integer_indexing(X, n_windows, window_length,
                                                   rng)


def pick_windows(array, window_length, offset='all'):
    """
    Pick windows independently for each row of a 2D array

    Parameters
    ----------
    array (numpy.ndarray):
        array whose rows are to be windowed
    window_length (int):
        Length of the window
    offset (numpy.ndarray or 'all'):
        offset[i] has the offsets, or starting indexes, for the
        windows that are to be extracted from array[i]. If 'all', return all
        possible windows.

    Returns
    -------
    windows (numpy.ndarray):
        array with the windows

    Shapes
    ------
    array: (n_rows, n_cols):
    offset: (n_rows, n_offsets)
    windows:    (n_rows, n_offsets, window_length)
    """

    n_rows, n_cols = array.shape

    if isinstance(offset, str) and offset == 'all':
        n_offsets = n_cols - window_length + 1
        offset = np.arange(n_offsets).reshape((1, n_offsets))
        offset = np.repeat(offset, n_rows, axis=0)

    if offset.ndim == 1: # Only one offset per row
        offset = np.expand_dims(offset, axis=1) # (n_rows,)->(n_rows,1)
        col_id = np.arange(window_length).reshape((1, window_length))
        row_id = np.arange(n_rows).reshape(n_rows, 1)
    elif offset.ndim == 2:
        # (n_rows,n_offsets)->(n_rows,n_offsets,1):
        offset = np.expand_dims(offset, axis=2)
        col_id = np.arange(window_length).reshape((1, 1, window_length))
        row_id = np.arange(n_rows).reshape(n_rows, 1, 1)

    # After broadcasting, col_id.shape=(n_rows,n_offsets,window_length)
    col_id = col_id + offset

    return array[row_id, col_id]
