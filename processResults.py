import numpy as np
import os
import pickle as pkl

# set up variables
task = "Dataset701_Set1"
fold = "all"

local = True
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"


labels = {"Background": 0,
          "Spleen": 1,
          "Right Kidney": 2,
          "Left Kidney": 3,
          "Gallbladder": 4,
          "Esophagus": 5,
          "Liver": 6,
          "Stomach": 7,
          "Aorta": 8,
          "Inferior Vena Cava": 9,
          "Pancreas": 10,
          "Right Adrenal Gland": 11,
          "Left Adrenal Gland": 12,
          "Duodenum": 13,
          "Bladder": 14}

def main():
    datasets = ["Dataset701_Set1", "Dataset702_Set2", "Dataset703_Set3"]

    # lists to store the results for each dataset
    av_dice_men = []
    std_dice_men = []
    av_hd_men = []
    std_hd_men = []

    av_dice_women = []
    std_dice_women = []
    av_hd_women = []
    std_hd_women = []

    for ds in datasets:
        preds_dir = os.path.join(root_dir, "inference", ds, fold)
        f = open(os.path.join(preds_dir, "dice_and_hd.pkl"), "rb")
        metrics = pkl.load(f)
        f.close()

        # Dice
        av_dice_men.append(np.nanmean(metrics["dice_men"], axis=1))
        std_dice_men.append(np.nanstd(metrics["dice_men"], axis=1))
        av_dice_women.append(np.nanmean(metrics["dice_women"], axis=1))
        std_dice_women.append(np.nanstd(metrics["dice_women"], axis=1))

        # Hausdorff
        hd_men = np.squeeze(metrics["hd_men"])
        hd_women = np.squeeze(metrics["hd_women"])

        # Replace infs with nans so we can compute average using nanmean
        hd_men[hd_men==np.inf] = np.nan
        hd_women[hd_women==np.inf] = np.nan

        av_hd_men.append(np.nanmean(hd_men, axis=1))
        std_hd_men.append(np.nanstd(hd_men, axis=1))
        av_hd_women.append(np.nanmean(hd_women, axis=1))
        std_hd_women.append(np.nanstd(hd_women, axis=1))

    organs = list(labels.keys())
    n_channels = len(labels)

    # First print out the information for men
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} ({1:.3f}) & {2:.3f} ({3:.3f}) & {4:.3f} ({5:.3f})".format(av_dice_men[0][i],
                                                                                                std_dice_men[0][i],
                                                                                                av_dice_men[1][i],
                                                                                                std_dice_men[1][i],
                                                                                                av_dice_men[2][i],
                                                                                                std_dice_men[2][i]) + r" \\")

    print('')

    # Then print out the information for women
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} ({1:.3f}) & {2:.3f} ({3:.3f}) & {4:.3f} ({5:.3f})".format(av_dice_women[0][i],
                                                                                                std_dice_women[0][i],
                                                                                                av_dice_women[1][i],
                                                                                                std_dice_women[1][i],
                                                                                                av_dice_women[2][i],
                                                                                                std_dice_women[2][i]) + r" \\")

    print('')

    # First print out the information for men
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} & {1:.3f} & {2:.3f}".format(av_dice_men[0][i],
                                                                  av_dice_men[1][i],
                                                                  av_dice_men[2][i]) + r" \\")

    print('')

    # Then print out the information for women
    for i in range(n_channels):
        print(organs[i] + " & {0:.3f} & {1:.3f} & {2:.3f}".format(av_dice_women[0][i],
                                                                  av_dice_women[1][i],
                                                                  av_dice_women[2][i]) + r" \\")




if __name__ == "__main__":
    main()