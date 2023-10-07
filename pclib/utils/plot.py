import matplotlib.pyplot as plt
from pclib.nn.layers import PrecisionWeighted

def plot_stats(stats, model):
    extra_rows = 0
    height = 8
    if not model.symmetric:
        extra_rows += 1
        height += 2 
    if model.bias:
        extra_rows += 1
        height += 2
    fig, axs = plt.subplots(2 + extra_rows, 2, figsize=(8, height))
    for i in range(len(model.layers)):
        axs.flat[0].plot(stats['R_norms'][i], label=f"Layer {i}")
        axs.flat[1].plot(stats['E_mags'][i], label=f"Layer {i}")
        axs.flat[2].plot(stats['WeightTD_means'][i], label=f"Layer {i}")
        axs.flat[3].plot(stats['WeightTD_stds'][i], label=f"Layer {i}")
        idx = 4
        if not model.symmetric:
            axs.flat[idx].plot(stats['WeightBU_means'][i], label=f"Layer {i}")
            axs.flat[idx+1].plot(stats['WeightBU_stds'][i], label=f"Layer {i}")
            idx += 2
        if model.bias:
            axs.flat[idx].plot(stats['Bias_means'][i], label=f"Layer {i}")
            axs.flat[idx+1].plot(stats['Bias_stds'][i], label=f"Layer {i}")
            idx += 2
        # if isinstance(model.layers[i], PrecisionWeighted):
        #     axs.flat[idx].plot(stats)

    axs.flat[0].set_title(f"X_norms")
    axs.flat[1].set_title(f"E_mags")
    axs.flat[2].set_title(f"Weight_means")
    axs.flat[3].set_title(f"Weight_stds")
    axs.flat[0].legend()
    axs.flat[1].legend()
    axs.flat[2].legend()
    axs.flat[3].legend()
    idx = 4
    if not model.symmetric:
        axs.flat[idx].set_title(f"WeightBU_means")
        axs.flat[idx+1].set_title(f"WeightBU_stds")
        axs.flat[idx].legend()
        axs.flat[idx+1].legend()
        idx += 2
    if model.bias:
        axs.flat[idx].set_title(f"Bias_means")
        axs.flat[idx+1].set_title(f"Bias_stds")
        axs.flat[idx].legend()
        axs.flat[idx+1].legend()
        idx += 2
    plt.show()