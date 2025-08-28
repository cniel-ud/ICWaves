from argparse import Namespace
from typing import Optional


def _get_conversion_factor(
    feature_extractor: str, srate: float, window_length: Optional[float]
) -> dict[str, float]:
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
    if "bowav" in feature_extractor:
        if window_length is None:
            raise ValueError(
                "window_length must be specified when using bowav feature extractor"
            )
        conversion_factors["bowav"] = 1 / window_length

    # Add psd_autocorr if it's in feature_extractor
    if "psd_autocorr" in feature_extractor:
        conversion_factors["psd_autocorr"] = srate

    if not conversion_factors:
        raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

    return conversion_factors


def _validate_segment_length(segment: float, factor: float) -> None:
    """Validates if segment length is compatible with conversion factor."""
    if factor < 1 and int(segment * factor) < segment * factor != 0:
        raise ValueError(f"Segment length {segment} is not divisible by {1 / factor}")


def _process_list_of_segment_lengths(
    segment_lengths: list[float], conversion_factors: dict[str, float]
) -> list[dict[str, int]]:
    """Process list of segments with their conversion factors."""
    processed_segment_lengths = []
    for segment_length in segment_lengths:
        segment_dict = {}
        for extractor, factor in conversion_factors.items():
            if segment_length == -1:
                segment_dict[extractor] = None
                continue
            _validate_segment_length(segment_length, factor)
            segment_dict[extractor] = int(segment_length * factor)
        processed_segment_lengths.append(segment_dict)

    return processed_segment_lengths


def _check_segment_length(segment_length):
    if isinstance(segment_length, float):
        return [segment_length]
    elif isinstance(segment_length, list):
        return segment_length
    else:
        raise ValueError("segment_length must be a float or a list of floats")


def convert_segment_length(
    segment_length,
    feature_extractor: str,
    srate: float,
    window_length: Optional[float] = None,
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
    conversion_factors = _get_conversion_factor(feature_extractor, srate, window_length)

    segment_length = _check_segment_length(segment_length)
    return _process_list_of_segment_lengths(segment_length, conversion_factors)
