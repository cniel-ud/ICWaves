import hashlib
import shlex
from typing import Optional

from icwaves.argparser import create_argparser_all_params


def get_validation_segment_length_string(valseglen: int) -> str:
    valseglen = "None" if valseglen == -1 else str(valseglen)
    return valseglen


def get_cmmn_suffix(cmmn_filter: Optional[str]) -> str:
    return f"_cmmn-{cmmn_filter}"


def read_args_from_file(file_path):
    # Read the file
    with open(file_path, "r") as file:
        file_contents = file.read()

    # Split the file contents into a list of arguments
    # shlex.split properly handles spaces within arguments
    args_list = shlex.split(file_contents)

    return args_list


def parse_config_file_args(path_to_config_file, feature_extractor):
    args_list = read_args_from_file(path_to_config_file)
    all_params_parser = create_argparser_all_params(feature_extractor)
    args, _ = all_params_parser.parse_known_args(args_list)
    args.feature_extractor = feature_extractor
    return args


def _build_centroid_assignments_file(args):
    subj_str = list_to_base36(args.subj_ids)
    cmmn_suffix = get_cmmn_suffix(args.cmmn_filter)
    base_name = (
        f"k-{args.num_clusters}_P-{args.centroid_length}"
        f"_winLen-{args.window_length}_minPerIC-{args.minutes_per_ic}"
        f"_cbookMinPerIc-{args.codebook_minutes_per_ic}"
        f"_cbookICsPerSubj-{args.codebook_ics_per_subject}"
        f"_subj-{subj_str}{cmmn_suffix}"
    )
    file_name = f"{base_name}.npy"

    return file_name


def _build_ics_and_labels_file(args):
    subj_str = list_to_base36(args.subj_ids)
    # add suffix to signal that CMMN filter was applied
    cmmn_suffix = get_cmmn_suffix(args.cmmn_filter)
    base_name = f"minPerIC-{args.minutes_per_ic}_subj-{subj_str}{cmmn_suffix}"
    file_name = f"{base_name}.npz"

    return file_name


def _build_preprocessed_data_file(args):
    subj_str = list_to_base36(args.subj_ids)
    cmmn_suffix = get_cmmn_suffix(args.cmmn_filter)
    base_name = f"winLen-{args.window_length}_minPerIC-{args.minutes_per_ic}_subj-{subj_str}{cmmn_suffix}"
    file_name = f"{base_name}.npz"

    return file_name


def build_base_classifier_name(args):

    valseglen = get_validation_segment_length_string(
        int(args.validation_segment_length)
    )
    cmmn_suffix = get_cmmn_suffix(args.cmmn_filter)
    idf_str = "_idf" if "bowav" in args.feature_extractor and args.use_idf else ""

    base_clf_name = f"{args.classifier_type}_{args.feature_extractor}_valSegLen{valseglen}{cmmn_suffix}{idf_str}"

    return base_clf_name


def list_to_base36(int_list):
    """Encode a list of subject ids into a short string for cache file names.

    Ids 1..34 are encoded as a 34-bit mask in base 36; other ids are ignored by
    the mask. In particular, the mask cannot tell apart EPIC's 0..909 from
    1..909, so when the list contains an id < 1 a suffix with the range and the
    number of ids is appended (plus a hash of the ids if they are not
    contiguous). Lists without ids < 1 keep their original names, so existing
    cache files (e.g. Emotion Study 8..35, EPIC 1..909) are still found.
    NOTE: ids > 34 are still ignored by the mask (kept for compatibility).
    """
    int_list = sorted(set(int(i) for i in int_list))
    # Create a 34-bit binary representation
    binary_str = ["0"] * 34
    for i in int_list:
        if 1 <= i <= 34:
            binary_str[i - 1] = "1"
    binary_str = "".join(binary_str)

    # Convert binary string to integer
    integer_representation = int(binary_str, 2)

    # Convert integer to base-36
    base36_representation = base36_encode(integer_representation)

    if int_list and int_list[0] < 1:
        lo, hi = int_list[0], int_list[-1]
        base36_representation += f"-{lo}to{hi}n{len(int_list)}"
        if len(int_list) != hi - lo + 1:
            digest = hashlib.md5(",".join(map(str, int_list)).encode()).hexdigest()
            base36_representation += f"-{digest[:8]}"

    return base36_representation


def base36_encode(number):
    assert number >= 0, "Number must be non-negative."
    if number == 0:
        return "0"

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base36 = ""
    while number:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36

    return base36
