import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def save_show_plot(filepath, neptune_run=None):
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')
        if neptune_run is not None:
            neptune_run["plots/" + filepath.name].upload(plt.gcf())
        plt.close()
    else:
        plt.show()


def plot_images_grid(x_array, filepath=None, neptune_run=None, cols=None, outlines=True):
    """ Input: matrix of images of shape: (rows, cols, h, w, d)
        saves image locally if filepath (extension included) is given, also to neptune if neptune_run is given,
        otherwise uses plt.show()"""
    x_rows, x_cols, height, width, depth = x_array.shape
    assert depth == 1 or depth == 3, "x_array must contain greyscale or RGB images"
    cols = (cols if cols else x_cols)
    rows = x_rows * int(np.ceil(x_cols / cols))

    plt.figure(figsize=(cols * 2, rows * 2))

    def draw_subplot(x_, ax_):
        if depth == 1:
            plt.imshow(x_.reshape([height, width]), cmap="Greys_r", vmin=0, vmax=1)
        elif depth == 3:
            plt.imshow(x_)
        if outlines:
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)
        else:
            ax_.set_axis_off()

    for j, x_row in enumerate(x_array):
        for i, x in enumerate(x_row[:x_cols], 1):
            # display original
            ax = plt.subplot(rows, cols, i + j * cols * int(rows / len(x_array)))  # rows, cols, subplot numbered from 1
            draw_subplot(x, ax)

    save_show_plot(filepath, neptune_run)


def yiq_to_rgb(yiq):
    conv_matrix = np.array([[1., 0.956, 0.619],
                            [1., -0.272, 0.647],
                            [1., -1.106, 1.703]])
    return np.tensordot(yiq, conv_matrix, axes=((-1,), (-1)))


def yiq_embedding(theta, phi):
    result = np.zeros(theta.shape + (3,))
    steps = 12
    rounding = True
    if rounding:
        theta_rounded = 2 * np.pi * np.round(steps * theta / (2 * np.pi)) / steps
        phi_rounded = 2 * np.pi * np.round(steps * phi / (2 * np.pi)) / steps
        theta = theta_rounded
        phi = phi_rounded
    result[..., 0] = 0.5 + 0.14 * np.cos((theta + phi) * steps / 2) - 0.2 * np.sin(phi)
    result[..., 1] = 0.25 * np.cos(phi)
    result[..., 2] = 0.25 * np.sin(phi)
    return yiq_to_rgb(result)


def plot_torus_angles(encoded_horizontal_angle, encoded_vertical_angle, colors, filepath=None, neptune_run=None,
                      xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi)):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.scatter(encoded_horizontal_angle, encoded_vertical_angle, color=colors)
    ax.set_title("Torus encoded")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    save_show_plot(filepath, neptune_run)


def density_histogram(neg, pos, filepath=None, neptune_run=None, bins=30, alpha=0.4):
    plt.figure()
    plt.hist(neg, bins=bins, density=True, histtype="bar", color="g", alpha=alpha)
    plt.hist(neg, bins=bins, density=True, histtype="step", color="g", alpha=1)
    plt.hist(pos, bins=bins, density=True, histtype="bar", color="r", alpha=alpha)
    plt.hist(pos, bins=bins, density=True, histtype="step", color="r", alpha=1)
    # plt.xlabel("ELBO value")
    plt.ylabel("Density")
    save_show_plot(filepath, neptune_run)


def density_histograms(values_list, names, filepath=None, neptune_run=None, bins=30, alpha=0.4):
    assert len(values_list) == len(names), "values_list and names must have the same length"
    plt.figure()
    for i, values in enumerate(values_list):
        color = sns.color_palette()[i]
        plt.hist(values, bins=bins, density=True, histtype="bar", color=color, alpha=alpha)
        plt.hist(values, bins=bins, density=True, histtype="step", color=color, alpha=1)
    plt.legend(labels=names)
    # plt.xlabel("ELBO value")
    plt.ylabel("Density")
    save_show_plot(filepath, neptune_run)


def density_plot(neg, pos, filepath=None, neptune_run=None):
    plt.figure()
    sns.kdeplot(neg, shade=True, color="g")
    sns.kdeplot(pos, shade=True, color="r")
    # plt.xlabel("ELBO value")
    plt.ylabel("Density")
    save_show_plot(filepath, neptune_run)


def density_plots(values_list, names, filepath=None, neptune_run=None):
    assert len(values_list) == len(names), "values_list and names must have the same length"
    plt.figure()
    for values in values_list:
        sns.kdeplot(values, shade=True)
    plt.legend(labels=names)
    # plt.xlabel("ELBO value")
    plt.ylabel("Density")
    save_show_plot(filepath, neptune_run)


def roc_pr_curves(neg, pos, filepath_roc=None, filepath_pr=None, neptune_run=None, return_fp_fn=False):
    # translate to scores from 0 to 1, where 1 means anomaly/pos (low ELBO) and 0 means normal/neg (high ELBO)
    min_val = min(np.min(neg), np.min(pos))
    max_val = max(np.max(neg), np.max(pos))
    y_true_neg = np.zeros(neg.shape)
    y_true_pos = np.ones(pos.shape)
    y_true = np.concatenate((y_true_neg, y_true_pos))
    y_score = (max_val - np.concatenate((neg, pos))) / (max_val - min_val)

    # get ROC curve values
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    auroc = auc(fpr, tpr)

    # plot ROC curve
    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    save_show_plot(filepath_roc, neptune_run)

    # get Precision-Recall values
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    # plot Precision-Recall curve
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [no_skill, no_skill], color="navy", lw=lw, linestyle='--')
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label="PR curve (AUC= %0.2f)" % auprc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    save_show_plot(filepath_pr, neptune_run)

    if return_fp_fn:
        # get threshold for optimal TPR - FPR, return indices of false positives (in neg) and false negatives (in pos)
        idx_best = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[idx_best]  # for the normalised and flipped values
        optimal_threshold = - optimal_threshold * (max_val - min_val) + max_val  # for the original values in pos & neg
        fp_indices = np.squeeze(np.argwhere(neg < optimal_threshold), axis=1)
        fn_indices = np.squeeze(np.argwhere(pos > optimal_threshold), axis=1)
        return auroc, auprc, fp_indices, fn_indices
    else:
        return auroc, auprc
