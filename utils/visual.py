import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import functional as F

def show_imgs(imgs, ax=None):
    if ax is None:
        ax = plt.gca()
    # imgs: R x C x H x W
    imgs = F.pad(imgs, [1, 1, 1, 1], mode='constant', value=255)  # padding is lrtb
    R, C, H, W = imgs.shape
    imgs = imgs.transpose(1, 2)  # R x H x C x W
    imgs = imgs.reshape(R * H, C * W)  # RH x CW

    ax.imshow(imgs, cmap='gray')
    ax.axis('off')
    return ax

def show_history_csv(csv_path, metrics, smoothing=0.0):
    '''Plots the metrics from a CSV file.
    
    - 'metrics' should be a nested list of metric names
    - If 'metrics' is a string, it is assumed to be a single metric
    - If 'metrics' is a list of strings, it is assumed to be multiple metrics on a single plot
    - Otherwise, 'metrics' should be a list of lists, where each inner list is a group of metrics to be plotted on a single plot
    '''
    df = pd.read_csv(csv_path)
    
    if isinstance(metrics, str):
        metrics = [[[metrics]]]  # 'accuracy' -> [[['accuracy']]]
    if isinstance(metrics, (list, tuple)) and isinstance(metrics[0], str):
        # ['accuracy'] -> [[['accuracy']]]
        # ['accuracy', 'loss'] -> [[['accuracy', 'loss']]]
        metrics = [[metrics]]
    if isinstance(metrics, (list, tuple)) and isinstance(metrics[0], (list, tuple)) and isinstance(metrics[0][0], str):
        # [['accuracy'], ['loss']] -> [[['accuracy']], [['loss']]]
        metrics = [metrics]

    # [['accuracy'], ['loss']] -> [[['accuracy']], [['loss']]]
    rows = len(metrics)
    cols = len(metrics[0])
    
    print(rows, cols, metrics)

    fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), sharex=True, squeeze=False)
    for row_idx, row in enumerate(metrics):
        for col_idx, group in enumerate(row):
            for metric in group:
                metric_mask = df[metric].notnull()
                x_data = df['epoch'][metric_mask]
                y_data = df[metric][metric_mask]
                if smoothing > 0.0:
                    y_data = y_data.ewm(alpha=1.0-smoothing).mean()
                ax[row_idx, col_idx].plot(x_data, y_data, label=metric)
            ax[row_idx, col_idx].legend()
            # if col_idx == 0:
            #     ax[row_idx, col_idx].set_ylabel('Accuracy')
        if row_idx == rows - 1:
            ax[row_idx, col_idx].set_xlabel('Epoch')
    return fig, ax