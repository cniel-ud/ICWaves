from argparse import Namespace
from enum import Enum
from typing import Callable, Tuple
from .types import DataBundle
from ..factories import create_feature_extractor
from ..feature_extractors.bowav import build_or_load_centroid_assignments_and_labels
from ..preprocessing import load_or_build_ics_and_labels
import logging


VALID_FEATURE_TYPES = ["bowav", "psd_autocorr", "bowav_psd_autocorr"]


def load_data_bundles(args: Namespace) -> dict[str, DataBundle]:
    """Load required data bundles based on feature extractor type."""
    data_bundles = {}

    if "bowav" in args.feature_extractor:
        logging.info("Building or loading centroid assignments and labels")
        data_bundles["bowav"] = build_or_load_centroid_assignments_and_labels(args)

    if "psd_autocorr" in args.feature_extractor:
        logging.info("Loading or building ICs and labels")
        data_bundles["psd_autocorr"] = load_or_build_ics_and_labels(args)

    return data_bundles


def get_feature_extractor_params(
    feature_type: str, data_bundles: dict[str, DataBundle]
) -> dict:
    """Get parameters for feature extractor creation."""
    params = {}

    if "bowav" in feature_type:
        params["n_centroids"] = data_bundles["bowav"].n_centroids
    if "psd_autocorr" in feature_type:
        params["srate"] = data_bundles["psd_autocorr"].srate

    return params


def get_feature_extractor(
    feature_type, data_bundles: dict[str, DataBundle]
) -> Tuple[dict[str, DataBundle], Callable]:
    """Load data based on feature extractor type.

    Args:
        feature_type: Feature extractor type.
        data_bundles: Dictionary of data bundles.

    Returns:
        Tuple of (data bundles dict, feature extractor callable)
    """
    if feature_type not in VALID_FEATURE_TYPES:
        raise ValueError(f"Invalid feature extractor type: {feature_type}")

    params = get_feature_extractor_params(feature_type, data_bundles)

    logging.info(f"Getting {feature_type} feature extractor...")
    feature_extractor = create_feature_extractor(feature_type, **params)

    return feature_extractor
