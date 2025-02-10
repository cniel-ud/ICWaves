from argparse import Namespace
from typing import Dict, List


def _get_conversion_factor(args: Namespace, srate: float) -> Dict[str, float | None]:
    """
    Computes factor to be multiplied with segment length (in seconds) to get:
        - number of windows for bowav feature
        - number of samples for psd_autocorr feature

    Args:
        args: Arguments containing feature extractor type and other parameters.
        srate: Sampling rate of the data.

    Returns:
        Conversion factor to be multiplied with segment length to get number of samples.
    """
    # Validate feature extractors upfront
    valid_extractors = {"bowav", "psd_autocorr"}
    if not any(extractor in args.feature_extractor for extractor in valid_extractors):
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")

    # Create conversion factors
    return {
        "bowav": 1 / args.window_length if "bowav" in args.feature_extractor else None,
        "psd_autocorr": srate if "psd_autocorr" in args.feature_extractor else None,
    }


def calculate_segment_length(
    args: Namespace, srate: float, train: bool = True
) -> Dict[str, List[int | None]]:
    """Convert training segment lengths based on feature extractor type."""
    conversion_factors = _get_conversion_factor(args, srate)

    segment_lengths: Dict[str, List[int | None]] = {}

    for extractor, factor in conversion_factors.items():
        if factor is not None:  # Only process active extractors
            if train:
                segment_lengths[extractor] = [
                    int(segment * factor) for segment in args.training_segment_length
                ]
            else:
                if args.validation_segment_length == -1:
                    segment_lengths[extractor] = [None]
                else:
                    segment_lengths[extractor] = [
                        int(args.validation_segment_length * factor)
                    ]

    return segment_lengths
