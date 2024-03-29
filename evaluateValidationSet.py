import numpy as np
import nibabel as nib
import os
import pickle as pkl
from monai.metrics import compute_hausdorff_distance
import argparse

from plotting import plotVolumeDistribution

# argparse
parser = argparse.ArgumentParser(description="Just an example",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--task", default="Dataset701_Set1", help="Task to evaluate")
args = vars(parser.parse_args())

# set up variables
task = args["task"]

local = True
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"

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

n_channels = int(len(labels))


def getVolume(pred, gt):
    # Get the organ volumes given the ground truth mask and the validation mask
    vol_preds = []
    vol_gts = []
    for channel in range(n_channels):
        vol_preds.append(np.sum(pred[pred == channel]))
        vol_gts.append(np.sum(gt[gt == channel]))

    return np.array(vol_preds), np.array(vol_gts)


def oneHotEncode(array):
    array_dims = len(array.shape)
    array_max = 15
    one_hot = np.zeros((array_max + 1, array.shape[0], array.shape[1], array.shape[2]))

    for i in range(0, array_max + 1):
        one_hot[i, :, :, :][array==i] = 1

    return one_hot


def computeHDDIstance(pred, gt):
    # To use the MONAI function pred must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
    # The values should be binarized.
    # gt: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
    # The values should be binarized.

    # Convert to one hot
    # covert predictions to one hot encoding
    pred_one_hot = oneHotEncode(pred)
    gt_one_hot = oneHotEncode(gt)

    # expand the number of dimensions to include batch
    pred_one_hot = np.expand_dims(pred_one_hot, axis=0)
    gt_one_hot = np.expand_dims(gt_one_hot, axis=0)

    hd = compute_hausdorff_distance(pred_one_hot, gt_one_hot, include_background=False, distance_metric='euclidean', percentile=None,
                               directed=False, spacing=None)

    return hd


def multiChannelDice(pred, gt, channels):

    dice = []

    for channel in range(channels):
        a = np.zeros(pred.shape)
        a[pred == channel] = 1

        b = np.zeros(gt.shape)
        b[gt == channel] = 1

        dice.append(np.sum(a[b == 1])*2.0 / (np.sum(a) + np.sum(a)))

    return np.array(dice)


def calculateMetrics():
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

    hd_men = []
    hd_women = []

    vol_pred_men = []
    vol_pred_women = []

    vol_gt_men = []
    vol_gt_women = []

    cases = os.listdir(preds_dir)
    for case in cases:
        if case.endswith(".nii.gz"):
            print(case)

            pred = nib.load(os.path.join(preds_dir, case)).get_fdata()
            gt = nib.load(os.path.join(gt_dir, case)).get_fdata()

            if np.unique(gt).sum() == 0:
                print("Only background")

            # Get Dice and NSD and volumes
            dice = multiChannelDice(pred, gt, n_channels)

            hd = computeHDDIstance(pred, gt)

            vol_pred, vol_gt = getVolume(pred, gt)

            if int(case[5:9]) in idx_women:
                dice_women.append(dice)
                hd_women.append(hd)
                vol_pred_women.append(vol_pred)
                vol_gt_women.append(vol_gt)
            elif int(case[5:9]) in idx_men:
                dice_men.append(dice)
                hd_men.append(hd)
                vol_pred_men.append(vol_pred)
                vol_gt_men.append(vol_gt)
            else:
                print("Not in list")

    print("Number of men: {}".format(len(dice_men)))
    print("Number of women: {}".format(len(dice_women)))

    dice_men = np.array(dice_men)
    dice_women = np.array(dice_women)

    #hd_men = np.array(hd_men)
    #hd_women = np.array(hd_women)

    vol_pred_men = np.array(vol_pred_men)
    vol_pred_women = np.array(vol_pred_women)

    vol_gt_men = np.array(vol_gt_men)
    vol_gt_women = np.array(vol_gt_women)

    f = open(os.path.join(preds_dir, "dice_and_hd.pkl"), "wb")
    pkl.dump({"dice_men": dice_men,
              "dice_women": dice_women,
              "hd_men": hd_men,
              "hd_women": hd_women,
              "vol_pred_men": vol_pred_men,
              "vol_pred_women": vol_pred_women,
              "vol_gt_women": vol_gt_women,
              "vol_gt_men": vol_gt_men}, f)
    f.close()


def main():
    #calculateMetrics()

    f = open(os.path.join(preds_dir, "dice_and_hd.pkl"), "rb")
    metrics = pkl.load(f)
    f.close()

    dice_men = metrics["dice_men"]
    dice_women = metrics["dice_women"]
    hd_men = metrics["hd_men"]
    hd_women = metrics["hd_women"]
    vol_pred_men = metrics["vol_pred_men"]
    vol_pred_women = metrics["vol_pred_women"]
    vol_gt_women = metrics["vol_gt_women"]
    vol_gt_men = metrics["vol_gt_men"]

    # cycle over each organ
    organs = list(labels.keys())

    for i in range(1, n_channels):
        organ = organs[i]

        if organ == "prostate/uterus":
            organ = "prostate or uterus"

        volumes_men = vol_gt_men[:, i]
        volumes_women = vol_gt_women[:, i]

        save_path = os.path.join(root_dir, "plots", "{}_volume.png".format(organ))

        plotVolumeDistribution(volumes_men, volumes_women, organ, save_path)






if __name__ == "__main__":
    main()