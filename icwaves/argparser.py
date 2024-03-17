from argparse import ArgumentParser


def create_argparser_one_parameter_one_split():
    parser = ArgumentParser()
    parser.add_argument(
        "--path-config-file",
        help="Path to config file with all the parameters",
        required=True,
    )
    parser.add_argument(
        "--job-id",
        type=int,
        help=(
            "ID of SLURM array job. It takes values in [0, n_params*n_subjects-1]"
            "where n_params is the number of parameters in the config file and "
            "n_subjects is the number of subjects."
        ),
    )

    return parser


def create_argparser_all_params():
    parser = ArgumentParser()
    parser.add_argument("--path-to-raw-data", help="Path to raw data", required=True)
    parser.add_argument(
        "--path-to-preprocessed-data", help="Path to preprocessed data", required=True
    )
    parser.add_argument(
        "--path-to-centroid-assignments",
        help="Path to centroid assignments",
        required=True,
    )
    parser.add_argument("--path-to-results", help="Path to results", required=True)
    parser.add_argument("--path-to-codebooks", help="Path to codebooks", required=True)
    parser.add_argument(
        "--subj-ids",
        type=int,
        help="A list with the subject ids to be used during training.",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--centroid-length", type=float, default=1.0, help="Centroid length in seconds"
    )
    # TODO: change type from float to int to arguments that are in seconds
    parser.add_argument(
        "--window-length",
        type=float,
        default=1.5,
        help="Length of window assigned to centroid, in seconds",
    )
    parser.add_argument(
        "--training-segment-length",
        type=float,
        nargs="+",
        default=[10, 30, 90, 180, 300],
        help="Length in seconds of segment used during training.",
    )
    parser.add_argument(
        "--validation-segment-length",
        type=int,
        default=300,
        help="Length in seconds of segment used during validation. Use -1 if you want to use all the time series.",
    )
    parser.add_argument(
        "--num-clusters", type=int, default=128, help="Number of clusters"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Value for n_jobs (sklearn)"
    )
    parser.add_argument(
        "--minutes-per-ic",
        type=float,
        default=50,
        help="Number of minutes per IC to extract BagOfWaves features",
    )
    parser.add_argument(
        "--regularization-factor",
        type=float,
        nargs="+",
        default=[0.1, 1, 10],
        help="Regularization factor used by the classifier. In LogisticRegression, it is the value of C.",
    )
    parser.add_argument(
        "--expert-weight",
        type=float,
        nargs="+",
        default=[1, 2, 4],
        help="Sample weight given to ICs with expert labels.",
    )
    parser.add_argument(
        "--l1-ratio", type=float, nargs="+", default=[0, 0.2, 0.4, 0.6, 0.8, 1]
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--penalty", default="elasticnet", choices=["l1", "l2", "elasticnet", "none"]
    )
    parser.add_argument(
        "--codebook-minutes-per-ic",
        type=float,
        default=50,
        help="Number of minutes per IC to train the class-specific codebook",
    )
    parser.add_argument(
        "--codebook-ics-per-subject",
        type=int,
        default=2,
        help="Maximum number of ICs per subject to train the class-specific codebook",
    )
    parser.add_argument(
        "--bowav-norm",
        help="Instance-wise normalization in BoWav",
        nargs="+",
        default=["none", "l_1", "l_2", "l_inf"],
    )

    return parser
