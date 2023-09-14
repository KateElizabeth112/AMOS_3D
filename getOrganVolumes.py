# Get the volumes of the organs for the whole dataset and look for statistical differences based on characteristics
# in the metadata
import os
import nibabel as nib
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

local = False

if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/rds/general/user/kc2322/home/data/AMOS_3D"

gt_seg_dir = os.path.join(root_dir, "nnUNet_raw", "Dataset200_AMOS", "labelsTr")
meta_data_path = os.path.join(root_dir, "labeled_data_meta_0000_0599.csv")

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


def calculate_volumes():
    # create containers to store the volumes
    volumes_f = []
    volumes_m = []

    # get a list of the files in the gt seg folder
    f_names = os.listdir(gt_seg_dir)

    # open the metadata
    meta = pd.read_csv(meta_data_path)
    ids_m = np.array(meta[meta["Patient's Sex"] == "M"]["amos_id"].values)
    ids_f = np.array(meta[meta["Patient's Sex"] == "F"]["amos_id"].values)

    for f in f_names:
        if f.endswith(".nii.gz"):
            # load image
            gt_nii = nib.load(os.path.join(gt_seg_dir, f))

            # get the volume of 1 voxel in mm3
            sx, sy, sz = gt_nii.header.get_zooms()
            volume = sx * sy * sz

            # find the number of voxels per organ in the ground truth image
            gt = gt_nii.get_fdata()
            volumes = []

            # cycle over each organ
            organs = list(labels.keys())

            for i in range(1, len(labels)):
                organ = organs[i]
                voxel_count = np.sum(gt == i)
                volumes.append(voxel_count * volume)

            # work out if the candidate is male or female
            subject = int(f[5:9])

            if subject in ids_f:
                volumes_f.append(np.array(volumes))
            elif subject in ids_m:
                volumes_m.append(np.array(volumes))
            else:
                print("Can't find subject in metadata list.")

    # Save the volumes ready for further processing
    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "wb")
    pkl.dump([np.array(volumes_m), np.array(volumes_f)], f)
    f.close()


def plotVolumes():
    f = open(os.path.join(root_dir, "volumes_gender.pkl"), "rb")
    [volumes_m, volumes_f] = pkl.load(f)
    f.close()

    # For each organ, plot the volume distributions
    organs = list(labels.keys())

    for i in range(1, len(labels)):
        organ = organs[i]
        plt.clf()
        plt.hist(volumes_m[:, i], label="Male")
        plt.hist(volumes_f[:, i], label="Female")
        plt.legend()
        plt.title(organ)
        plt.show()




def main():
    calculate_volumes()
    #plotVolumes()



if __name__ == "__main__":
    main()