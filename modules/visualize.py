import numpy as np
import matplotlib.pyplot as plt

def visualize_ecg(normalized_signal, sampling_frequency=400, title="Normalized ECG Signals", lead_names=None):
    """
    Visualize normalized ECG signals from a 2D numpy array.

    Each lead is plotted in its own subplot for clarity.

    Args:
        normalized_signal (numpy.ndarray): A 2D numpy array where rows are samples
                                           and columns are leads. Shape: (samples, leads).
        sampling_frequency (int, optional): The sampling frequency of the signal in Hz.
                                            Defaults to 400.
        title (str, optional): The title for the entire figure.
                               Defaults to "Normalized ECG Signals".
        lead_names (list of str, optional): A list of names for each lead. If None,
                                            leads will be named 'Lead 1', 'Lead 2', etc.
    """
    if not isinstance(normalized_signal, np.ndarray) or normalized_signal.ndim != 2:
        raise ValueError("Input signal must be a 2D numpy array.")

    num_samples, num_leads = normalized_signal.shape
    time_axis = np.arange(num_samples) / sampling_frequency

    if lead_names and len(lead_names) != num_leads:
        raise ValueError("Number of lead names must match number of leads in the signal.")

    if not lead_names:
        lead_names = [f'Lead {i+1}' for i in range(num_leads)]

    fig, axes = plt.subplots(num_leads, 1, figsize=(15, 2 * num_leads), sharex=True, sharey=True)
    if num_leads == 1:
        axes = [axes] # make it iterable

    for i in range(num_leads):
        axes[i].plot(time_axis, normalized_signal[:, i])
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle and xlabel
    plt.savefig(f"visualizations/{title.replace(' ', '_')}.png")
