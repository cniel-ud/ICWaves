from pathlib import Path
from time import perf_counter
import logging

import numpy as np
from numpy.random import default_rng

from icwaves.argparser import (
    create_argparser_train_dict,
    create_argparser_train_dict_config,
)
from icwaves.data_loaders import load_raw_train_set_per_class
from icwaves.file_utils import read_args_from_file
from icwaves.sikmeans.shift_kmeans import shift_invariant_k_means

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting dictionary training process")
    
    parser = create_argparser_train_dict_config()
    main_args = parser.parse_args()
    n_jobs = main_args.n_jobs
    
    args_list = read_args_from_file(main_args.path_to_config_file)
    parser = create_argparser_train_dict()
    args = parser.parse_args(args_list)

    rng = default_rng(13)

    X, srate = load_raw_train_set_per_class(args, rng)
    logger.info(f"Loaded data with shape {X.shape}, sampling rate {srate} Hz")

    centroid_length = int(args.centroid_length * srate)

    metric, init = "cosine", "random"
    logger.info(f"Running sikmeans with {args.num_clusters} clusters")
    
    t_start = perf_counter()
    centroids, labels, shifts, distances, inertia, _ = shift_invariant_k_means(
        X,
        args.num_clusters,
        centroid_length,
        metric=metric,
        init=init,
        n_init=args.n_runs,
        rng=rng,
        verbose=True,
        n_jobs=n_jobs,
    )
    t_stop = perf_counter()
    logger.info(f"Time running sikmeans: {t_stop-t_start:.3f} seconds")

    results_dir = Path(args.path_to_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = results_dir.joinpath(
        f"sikmeans_P-{args.centroid_len}_k-{args.num_clusters}"
        f"_class-{args.class_label}_minutesPerIC-{args.minutes_per_ic}"
        f"_icsPerSubj-{args.ics_per_subject}.npz"
    )
    logger.info(f"Saving results to: {out_file}")
    
    with out_file.open("wb") as f:
        np.savez(
            out_file,
            centroids=centroids,
            labels=labels,
            shifts=shifts,
            distances=distances,
            inertia=inertia,
        )
    
    logger.info("Dictionary training completed")