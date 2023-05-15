# Convert the names of the training and test images to have the nnUNet format
import os

#ROOT_DIR = '/Users/katecevora/Documents/PhD/data/AMOS_3D/'
ROOT_DIR = '/vol/biomedic3/kc2322/data/AMOS_3D'
TRAIN_DIR = os.path.join(ROOT_DIR, "nnUNet_raw/Dataset200_AMOS/imagesTr")


def main():
    # list files in training directory
    names = os.listdir(TRAIN_DIR)

    for n in names:
        if n.endswith(".nii.gz"):
            if not (n.split(".")[0][-1] == '0'):
                root = n.split(".")[0]
                n_new = root + "_0000.nii.gz"

                os.rename(os.path.join(TRAIN_DIR, n), os.path.join(TRAIN_DIR, n_new))


if __name__ == "__main__":
    main()