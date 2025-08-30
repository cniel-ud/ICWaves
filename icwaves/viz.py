from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path


def plot_confusion_matrix(
    cm,
    cmap,
    display_labels=None,
    values_format=None,
    colorbar=True,
    ylabel=None,
    xticks_rotation="horizontal",
):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    n_rows, n_cols = cm.shape

    im_kw = dict(interpolation="nearest", cmap=cmap)

    im_ = ax.imshow(cm, **im_kw, vmin=0.0, vmax=1.0)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    text_ = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_rows), range(n_cols)):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        if values_format is None:
            text_cm = format(cm[i, j], ".2g")
            if cm.dtype.kind != "f":
                text_d = format(cm[i, j], "d")
                if len(text_d) < len(text_cm):
                    text_cm = text_d
        else:
            text_cm = format(cm[i, j], values_format)

        text_[i, j] = ax.text(j, i, text_cm, ha="center", va="center", color=color)

    if display_labels is None:
        display_labels = [0] * 2
        display_labels[0] = np.arange(n_cols)
        display_labels[1] = np.arange(n_rows)

    if colorbar:
        fig.colorbar(im_, ax=ax)
    ax.set(
        xticks=np.arange(n_cols),
        yticks=np.arange(n_rows),
        xticklabels=display_labels[0],
        yticklabels=display_labels[1],
        ylabel=ylabel if ylabel is not None else "True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_rows - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    return fig, ax


def plot_line_with_error_area(
    ax, df, x, y, error, color="blue", add_std_label=False, label=None
):
    """
    Plot a line with error area.

    Args:
        ax: Matplotlib axis to plot on
        df: DataFrame containing data
        x: Column name for x-axis data
        y: Column name for y-axis data
        error: Column name for error data
        color: Color name or hex code
        add_std_label: Whether to add a separate label for the error area
        label: Custom label for the line (if None, uses the y column name)

    Returns:
        Updated matplotlib axis
    """
    # Use custom label if provided, otherwise use column name
    plot_label = label if label is not None else y

    # If color starts with 'tab:', use it directly, otherwise add the prefix
    if color.startswith("tab:"):
        color_str = color
    else:
        color_str = f"tab:{color}"

    ax.plot(
        df[x],
        df[y],
        label=plot_label,
        color=color_str,
    )

    std_label = error if add_std_label else None
    ax.fill_between(
        df[x],
        df[y] - df[error],
        df[y] + df[error],
        color=color_str,
        alpha=0.2,
        label=std_label,
    )
    return ax


def create_comparison_plot(
    results_df,
    fixed_params,
    vary_by,
    include_iclabel=True,
    figsize=(10, 6),
    save_path=None,
    add_title=True,
):
    """
    Create a comparison plot showing mean F1 score vs prediction window.

    This simplified function focuses specifically on plotting prediction window on the x-axis
    and mean F1 score on the y-axis, with one line for each value of the vary_by parameter.

    Args:
        results_df: DataFrame containing all results (including pre-computed ICLabel data)
        fixed_params: Dict of parameters to fix for filtering (e.g., {'eval_dataset': 'cue', 'validation_segment_len': 300})
        vary_by: Parameter to vary in the plot (e.g., 'feature_extractor', 'classifier_type')
        include_iclabel: Whether to include ICLabel results (must be pre-computed in results_df)
        figsize: Size of the figure
        save_path: Path to save the figure (if None, don't save)
        add_title: If True, add title. This is for debugging purposes.

    Returns:
        Figure and axis objects
    """
    # Define standard x ticks for log scale
    global_x_ticks = np.array([0.15, 0.5, 1, 2, 3, 5, 10, 30, 50])
    val_seg_len_map = {-1: "All", 300: "5-minutes"}

    # Filter the data based on fixed parameters
    filtered_df = results_df.copy()
    for param, value in fixed_params.items():
        if param != "cmmn_filter":  # Don't filter on cmmn_filter
            filtered_df = filtered_df[filtered_df[param] == value]

    # Validate that we have data after filtering
    if len(filtered_df) == 0:
        raise ValueError("No data matching the specified filters")

    # Get unique values for the parameter we're varying
    vary_values = filtered_df[vary_by].unique()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Add ICLabel reference if requested and available in the data
    if include_iclabel:
        # Get ICLabel data from the results DataFrame
        eval_dataset = fixed_params.get("eval_dataset")
        if eval_dataset:
            # Filter for ICLabel results for this dataset
            iclabel_data = results_df[
                (results_df["eval_dataset"] == eval_dataset)
                & (results_df["feature_extractor"] == "iclabel")
            ]

            if len(iclabel_data) > 0:
                # Sort by prediction window
                iclabel_data = iclabel_data.sort_values("prediction_window")

                # Plot ICLabel results
                ax = plot_line_with_error_area(
                    ax,
                    iclabel_data,
                    "prediction_window",
                    "mean_f1",
                    "std_f1",
                    color="red",
                    label="ICLabel",
                )

    # Create color mapping for the varying parameter
    vary_colors = plt.cm.tab10(np.linspace(0, 1, len(vary_values)))
    vary_color_map = {value: vary_colors[i] for i, value in enumerate(vary_values)}

    # Get unique filter values in the filtered dataframe
    filter_values = filtered_df["cmmn_filter"].unique()

    # Create a colormap specifically for filters with pastel colors
    filter_colors = plt.cm.Pastel1(np.linspace(0, 1, len(filter_values)))
    filter_color_map = {
        value: filter_colors[i] for i, value in enumerate(filter_values)
    }

    # Line styles for different filters
    filter_line_styles = {
        "None": "-",  # Solid line
        "normed-barycenter": "--",  # Dashed line
        "unnormed-barycenter": "-.",  # Dash-dot line
        "subj_to_subj": ":",  # Dotted line
    }

    # Plot for each combination of vary_by parameter and cmmn_filter
    for vary_value in vary_values:
        # Get data for this value of the varying parameter
        vary_data = filtered_df[filtered_df[vary_by] == vary_value]

        # Skip if no data
        if len(vary_data) == 0:
            continue

        # Process each filter for this vary_value
        for filter_value in filter_values:
            # Get data for this filter value
            value_data = vary_data[vary_data["cmmn_filter"] == filter_value]

            # Skip if no data for this combination
            if len(value_data) == 0:
                continue

            # Sort by prediction window
            value_data = value_data.sort_values("prediction_window")

            # Create label combining vary_by parameter and filter
            if vary_by == "feature_extractor":
                base_label = vary_value
            elif vary_by == "classifier_type":
                base_label = vary_value
            else:
                base_label = f"{vary_by}: {vary_value}"

            # Add filter to the label
            label = f"{base_label} (filter: {filter_value})"

            # Choose color based on the varying parameter
            color = vary_color_map[vary_value]

            # Choose line style based on filter
            line_style = filter_line_styles.get(filter_value, "-")

            # Plot the line with appropriate style
            ax.plot(
                value_data["prediction_window"],
                value_data["mean_f1"],
                label=label,
                color=color,
                linestyle=line_style,
            )

            # Add error area
            ax.fill_between(
                value_data["prediction_window"],
                value_data["mean_f1"] - value_data["std_f1"],
                value_data["mean_f1"] + value_data["std_f1"],
                color=color,
                alpha=0.1,  # Lower alpha for better visibility with multiple filters
            )

    # Format the plot
    ax.set_xscale("log")
    ax.set_xticks(global_x_ticks, labels=global_x_ticks)
    ax.set_xlim(filtered_df["prediction_window"].min(), 50)
    ax.set_xlabel("Prediction window [minutes]")
    ax.set_ylabel("Mean Brain F1 score")

    if add_title:
        # Create descriptive title
        title_parts = []
        for param, value in fixed_params.items():
            if param == "validation_segment_len":
                value = val_seg_len_map.get(value, value)
            if param != "cmmn_filter":  # Don't include cmmn_filter in title
                title_parts.append(f"{param}: {value}")

        ax.set_title(f"Mean Brain F1 score | {' | '.join(title_parts)}")
    ax.legend()
    ax.grid(True)

    # Save the figure if a path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
