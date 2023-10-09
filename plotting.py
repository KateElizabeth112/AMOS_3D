import matplotlib.pyplot as plt
import numpy as np

lblu = "#add9f4"
lred = "#f36860"


def plotVolumeDistribution(volumes_men, volumes_women, organ, save_path):
    # Plot the distribution of organ volumes for a single organ for men and women

    # First find the bins
    v_min_m = np.min(volumes_men)
    v_min_f = np.min(volumes_women)
    v_min = np.min((v_min_f, v_min_m))

    v_max_m = np.max(volumes_men)
    v_max_f = np.max(volumes_women)
    v_max = np.max((v_max_f, v_max_m))

    step = (v_max - v_min) / 8
    bins = np.arange(v_min, v_max + step, step)

    # Calculate averages to add to the plot
    v_av_men = np.mean(volumes_men)
    v_av_women = np.mean(volumes_women)

    plt.clf()
    plt.hist(volumes_men, color=lblu, alpha=0.6, label="Male", bins=bins)
    plt.axvline(x=v_av_men, color=lblu, label="Male average")
    plt.hist(volumes_women, color=lred, alpha=0.6, label="Female", bins=bins)
    plt.axvline(x=v_av_women, color=lred, label="Female average")
    plt.title(organ + " volume")
    plt.xlabel("Volume in voxels")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(save_path)


def plotDomainShiftAge():
    mu1 = 20
    mu2 = 50
    sig = 10

    # randomly draw some points from a distribution
    s1 = np.random.normal(mu1, sig, 500)
    s2 = np.random.normal(mu2, sig, 500)
    plt.hist(s1, 20, density=True, label="Domain 1", alpha=0.6, color=lred)
    plt.hist(s2, 20, density=True, label="Domain 2", alpha=0.6, color=lblu)
    plt.xlim(0, 100)
    plt.legend()
    plt.xlabel("Age")
    plt.yticks([])
    plt.ylabel("Frequency")
    plt.show()


def main():
    plotDomainShiftAge()


if __name__ == "__main__":
    main()
