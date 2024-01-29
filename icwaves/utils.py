from pathlib import Path


def _build_centroid_assignments_file(args):
    base_name = (
        f"k-{args.num_clusters}_P-{args.centroid_length}"
        f"_winlen-{args.window_length}_minPerIC-{args.minutes_per_ic}"
        f"_cbookMinPerIc-{args.codebook_minutes_per_ic}"
        f"_cbookICsPerSubj-{args.codebook_ics_per_subject}"
    )
    file_name = f"{base_name}.npy"
    data_folder = Path(args.path_to_centroid_assignments)
    data_folder.mkdir(exist_ok=True, parents=True)
    data_file = data_folder.joinpath(file_name)

    return data_file


def _build_preprocessed_data_file(args):
    base_name = f"winlen-{args.window_length}_minPerIC-{args.minutes_per_ic}"
    file_name = f"{base_name}.npz"
    data_folder = Path(args.path_to_preprocessed_data)
    data_folder.mkdir(exist_ok=True, parents=True)
    data_file = data_folder.joinpath(file_name)

    return data_file


def _build_results_file(
    args,
    regularization_factor,
    l1_ratio,
    expert_weight,
    bowav_norm,
    train_segment_length,
    validation_segment_length,
):
    centroid_assignment_base = _build_centroid_assignments_file(args)
    centroid_assignment_base = centroid_assignment_base.stem
    preprocessed_data_base = _build_preprocessed_data_file(args)
    preprocessed_data_base = preprocessed_data_base.stem

    C_str = "_".join([str(i) for i in regularization_factor])
    l1_ratio_str = "_".join([str(i) for i in l1_ratio])
    ew_str = "_".join([str(i) for i in expert_weight])
    train_segment_length_str = "_".join([str(i) for i in train_segment_length])
    validation_segment_length_str = "_".join(
        [str(i) for i in validation_segment_length]
    )
    bowav_norm_str = "_".join([str(i) for i in bowav_norm])

    classifier_base = (
        f"clf-lr_penalty-{args.penalty}_solver-saga_C-{C_str}"
        f"_l1_ratio-{l1_ratio_str}"
        f"_expert_weight-{ew_str}"
        f"_train_segment_length-{train_segment_length_str}"
        f"_validation_segment_length-{validation_segment_length_str}"
        f"_bowav_norm-{bowav_norm_str}"
    )
    classifier_fname = (
        f"{centroid_assignment_base}_{preprocessed_data_base}_"
        f"{classifier_base}.pickle"
    )
    results_folder = Path(args.path_to_results)
    results_folder.mkdir(exist_ok=True, parents=True)
    results_file = results_folder.joinpath(classifier_fname)

    return results_file
