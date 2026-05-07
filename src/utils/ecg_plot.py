import numpy as np
import matplotlib.pyplot as plt

colors = ["k", "c", "b", "m"]

f01 = 20
f02 = 16

def make_ecg_plot(
    ecg: np.ndarray, 
    ecg_duration: float,
    frequency: int,
    savename: str=None, 
    n_ecg: int=1
) -> None:
    """
    Args:
        ecg (np.ndarray): Array of size (data length,).
        ecg_duration (float): 
        frequency (int): 
        savename (str): Filename for saving.
    Returns:
        None
    """
    cm = 1 / 2.54

    # Define figure width.
    # 2.5 [cm / sec]: definition by ecg plot.
    # 2.5 [cm / sec] * `per_fig_length` [sec] 
    width = ecg_duration * 2.5

    x_scale_buffer = 10

    if n_ecg > 1:
        ecg_len = len(ecg[0])
        ecg_abs = [
            np.abs(ecg[i]).max() for i in range(n_ecg)
        ]
        ecg_abs = max(ecg_abs)
    else:
        ecg_len = len(ecg)
        ecg_abs = np.abs(ecg).max()

    # Define figure height.
    height = 4 * 2
    if ecg_abs > 2:
        y_min = -4
        y_max = 4
        major_y_ticks = np.arange(y_min, y_max+2, 2)
        minor_y_ticks = np.arange(y_min, y_max+0.2, 0.2)
        # height *= 2
    else:
        y_min = -2
        y_max = 2
        major_y_ticks = np.arange(y_min, y_max+1)
        minor_y_ticks = np.arange(y_min, y_max+0.1, 0.1)
    y_scale_buffer = 0.1

    fig_height = height * cm
    fig_width = width * cm

    fig = plt.figure(
        figsize=(fig_width, fig_height)
    )
    ax = fig.add_subplot(1, 1, 1)


    major_x_ticks = np.arange(ecg_len+1)[::frequency]
    minor_x_ticks = np.arange(ecg_len+1)[::frequency//10]

    ax.set_xticks(major_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)

    if n_ecg > 1:
        for ecg_idx in range(n_ecg):
            ax.plot(ecg[ecg_idx], color=colors[ecg_idx])
    else:
        ax.plot(ecg, color="k")

    x_labels = [
        str(t//frequency) if t % frequency == 0 else "" 
        for t in major_x_ticks
    ]
    ax.set_xticklabels(x_labels, fontsize=f02)
    ax.set_xlabel("[sec]", fontsize=f01)
    ax.set_xlim(-1 * x_scale_buffer, ecg_len + x_scale_buffer)

    # ax.set_ylabel("[mV]")
    ax.set_ylabel("scaled amplitude", fontsize=f01)
    ax.set_ylim(y_min - y_scale_buffer, y_max + y_scale_buffer)

    plt.grid(
        visible=True, 
        axis="both", 
        which="major", 
        color="y", 
        linestyle="-", 
        linewidth=1
    )
    plt.grid(
        visible=True, 
        axis="both", 
        which="minor", 
        color="y", 
        linestyle="dotted", 
        alpha=0.3, 
        linewidth=1
    )

    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
        plt.close()
    else:
        return fig