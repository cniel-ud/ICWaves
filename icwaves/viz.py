import numpy as np
from itertools import product

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

        text_[i, j] = ax.text(
            j, i, text_cm, ha="center", va="center", color=color
        )

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
