from argparse import Namespace
from typing import Optional


def _get_conversion_factor(args: Namespace, srate: float) -> dict[str, float]:
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
    conversion_factors = {}

    # Only add bowav if it's in feature_extractor and has window_length
    if "bowav" in args.feature_extractor:
        if not hasattr(args, "window_length"):
            raise ValueError(
                "window_length must be specified when using bowav feature extractor"
            )
        conversion_factors["bowav"] = 1 / args.window_length

    # Add psd_autocorr if it's in feature_extractor
    if "psd_autocorr" in args.feature_extractor:
        conversion_factors["psd_autocorr"] = srate

    if not conversion_factors:
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")

    return conversion_factors


def _validate_segment_length(segment: float, factor: float) -> None:
    """Validates if segment length is compatible with conversion factor."""
    if factor < 1 and int(segment * factor) < segment * factor != 0:
        raise ValueError(f"Segment length {segment} is not divisible by {1 / factor}")


def _process_list_of_segments(
    segments: list[float], conversion_factors: dict[str, float]
) -> list[dict[str, int]]:
    """Process list of segments with their conversion factors."""
    processed_segments = []
    for segment in segments:
        segment_dict = {}
        for extractor, factor in conversion_factors.items():
            _validate_segment_length(segment, factor)
            segment_dict[extractor] = int(segment * factor)
        processed_segments.append(segment_dict)

    return processed_segments


def _process_validation_segment(
    segment_length: float, conversion_factors: dict[str, float]
) -> list[dict[str, Optional[int]]]:
    """Process validation segment with conversion factors."""
    if segment_length == -1:
        return [{extractor: None for extractor in conversion_factors}]

    return _process_list_of_segments([segment_length], conversion_factors)


def calculate_segment_length(
    args: Namespace, srate: float, train: bool = True
) -> list[dict[str, Optional[int]]]:
    """
    Convert segment lengths based on feature extractor type.

    Args:
        args: Arguments containing feature extractor parameters
        srate: Sampling rate of the data
        train: Whether processing training or validation segments

    Returns:
        List of dictionaries mapping feature extractors to segment lengths
    """
    conversion_factors = _get_conversion_factor(args, srate)

    if train:
        return _process_list_of_segments(
            args.training_segment_length, conversion_factors
        )
    else:
        return _process_validation_segment(
            args.validation_segment_length, conversion_factors
        )
