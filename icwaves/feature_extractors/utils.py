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


def _validate_segment_length(segment: float, factor: float) -> None:
    """Validates if segment length is compatible with conversion factor."""
    if factor < 1 and segment % factor != 0:
        raise ValueError(
            f"Segment length {segment} is not divisible by conversion factor {factor}"
        )


def _process_training_segments(
    segments: list[float], conversion_factors: dict[str, float]
) -> list[dict[str, int]]:
    """Process training segments with their conversion factors."""
    result = []
    for segment in segments:
        for extractor, factor in conversion_factors.items():
            _validate_segment_length(segment, factor)
            result.append({extractor: int(segment * factor)})
    return result


def _process_validation_segment(
    segment_length: float, conversion_factors: dict[str, float]
) -> list[dict[str, Optional[int]]]:
    """Process validation segment with conversion factors."""
    if segment_length == -1:
        return [{extractor: None for extractor in conversion_factors}]

    result = []
    for extractor, factor in conversion_factors.items():
        _validate_segment_length(segment_length, factor)
        result.append({extractor: int(segment_length * factor)})
    return result


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
        return _process_training_segments(
            args.training_segment_length, conversion_factors
        )
    else:
        return _process_validation_segment(
            args.validation_segment_length, conversion_factors
        )
