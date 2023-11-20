from pathlib import Path

def _build_centroid_assignments_file(args):

    base_name = (
        f'k-{args.num_clusters}_P-{args.centroid_len}'
        f'_winlen-{args.window_len}_minPerIC-{args.minutes_per_ic}'
        f'_cbookMinPerIc-{args.codebook_minutes_per_ic}'
        f'_cbookICsPerSubj-{args.codebook_ics_per_subject}'
    )
    file_name = f'{base_name}.npy'
    data_folder = Path(args.path_to_centroid_assignments)
    data_folder.mkdir(exist_ok=True, parents=True)
    data_file = data_folder.joinpath(file_name)

    return data_file

def _build_preprocessed_data_file(args):

    base_name = (
        f'_winlen-{args.window_len}_minPerIC-{args.minutes_per_ic}'
    )
    file_name = f'{base_name}.npz'
    data_folder = Path(args.path_to_preprocessed_data)
    data_folder.mkdir(exist_ok=True, parents=True)
    data_file = data_folder.joinpath(file_name)

    return data_file