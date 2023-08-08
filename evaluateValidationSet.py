import numpy as np
import nibabel as nib
import os
import pickle as pkl

root_dir = "/vol/biomedic3/kc2322/data/AMOS_3D"
task = "Dataset702_Set2"
fold = "all"

preds_dir = os.path.join(root_dir, "inference", task, fold)
gt_dir = os.path.join(root_dir, "nnUNet_raw", task, "labelsTs")

labels = {"background": 0,
          "spleen": 1,
          "right kidney": 2,
          "left kidney": 3,
          "gallbladder": 4,
          "esophagus": 5,
          "liver": 6,
          "stomach": 7,
          "aorta": 8,
          "inferior vena cava": 9,
          "pancreas": 10,
          "right adrenal gland": 11,
          "left adrenal gland": 12,
          "duodenum": 13,
          "bladder": 14,
          "prostate/uterus": 15}

n_channels = 16

def multiChannelDice(pred, gt, channels):

    dice = []

    for channel in range(channels):
        a = np.zeros(pred.shape)
        a[pred == channel] = 1

        b = np.zeros(gt.shape)
        b[gt == channel] = 1

        dice.append(np.sum(a[b == 1])*2.0 / (np.sum(a) + np.sum(a)))

    return np.array(dice)


def main():
    # get a list of male and female IDs
    f = open(os.path.join(root_dir, "splits", "set1_splits.pkl"), "rb")
    set_1_ids = pkl.load(f)
    f.close()

    ids_ts = set_1_ids["test"]
    n_ids = len(ids_ts)
    idx_women = ids_ts[0:int(n_ids / 2)]
    idx_men = ids_ts[int(n_ids / 2):]

    dice_men = []
    dice_women = []

    cases = os.listdir(preds_dir)
    for case in cases:
        if case.endswith(".nii.gz"):
            print(case)

            pred = nib.load(os.path.join(preds_dir, case)).get_fdata()
            gt = nib.load(os.path.join(gt_dir, case)).get_fdata()

            if np.unique(gt).sum() == 0:
                print("Only background")

            # Get Dice and NSD
            dice = multiChannelDice(pred, gt, n_channels)

            if "s" + case[5:9] in idx_women:
                dice_women.append(dice)
            else:
                dice_men.append(dice)

    print("Number of men: {}".format(len(dice_men)))
    print("Number of women: {}".format(len(dice_women)))

    dice_men = np.array(dice_men)
    dice_women = np.array(dice_women)

    av_dice_men = np.nanmean(dice_men, axis=1)
    av_dice_women = np.nanmean(dice_women, axis=1)

    organs = labels.keys()
    for i in range(n_channels):
        print(organs[i] + ":\t {0:.3f} \t {1:.3f}".format(av_dice_men[i], av_dice_women[i]))


if __name__ == "__main__":
    main()