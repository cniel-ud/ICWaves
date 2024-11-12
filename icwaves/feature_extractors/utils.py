from argparse import Namespace
from typing import List


def _get_conversion_factor(args, srate):
    """
    Computes factor to be multiplied with segment length to get:
        - number of windows for BoWav feature
        - number of samples for psd_autocorr feature

    Args:
        args: Arguments containing feature extractor type and other parameters.
        srate: Sampling rate of the data.

    Returns:
        Conversion factor to be multiplied with segment length to get number of samples.
    """
    if args.feature_extractor == "bowav":
        conversion_factor = 1 / args.window_length
    elif args.feature_extractor == "psd_autocorr":
        conversion_factor = srate
    else:
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")
    return conversion_factor


def calculate_segment_length(
    args: Namespace, srate: float, train: bool = True
) -> List[int]:
    """Convert training segment lengths based on feature extractor type."""
    conversion_factor = _get_conversion_factor(args, srate)

    if train:
        return [int(s * conversion_factor) for s in args.training_segment_length]
    else:
        if args.validation_segment_length == -1:
            return [None]
        else:
            return [int(args.validation_segment_length * conversion_factor)]
