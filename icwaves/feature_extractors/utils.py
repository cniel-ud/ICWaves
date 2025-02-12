from argparse import Namespace
from typing import Dict, List, Optional


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
    conversion_factors = {
        "bowav": 1 / args.window_length,
        "psd_autocorr": srate,
    }
    if not any(
        extractor in args.feature_extractor for extractor in conversion_factors.keys()
    ):
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")

    # Create conversion factors
    return {
        extractor: factor
        for extractor, factor in conversion_factors.items()
        if extractor in args.feature_extractor
    }


def calculate_segment_length(
    args: Namespace, srate: float, train: bool = True
) -> Dict[str, List[int | None]]:
    """Convert training segment lengths based on feature extractor type."""
    conversion_factors = _get_conversion_factor(args, srate)

    segment_lengths: list[dict[str, Optional[int]]] = []

    if train:
        for segment in args.training_segment_length:
            segment_lengths.append(
                {
                    extractor: int(segment * factor)
                    for extractor, factor in conversion_factors.items()
                }
            )
    else:
        if args.validation_segment_length == -1:
            segment_lengths.append(
                {extractor: None for extractor in conversion_factors.keys()}
            )
        else:
            segment_lengths.append(
                {
                    extractor: int(args.validation_segment_length * factor)
                    for extractor, factor in conversion_factors.items()
                }
            )

    return segment_lengths
