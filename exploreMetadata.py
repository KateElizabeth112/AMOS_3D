# explore the AMOS metadata
import pandas as pd
import numpy as np
import os
import pickle as pkl
import shutil

local = True

if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
else:
    root_dir = "/vol/biomedic3/kc2322/data/AMOS_3D/"

input_folder = os.path.join(root_dir, "nnUNet_raw/Dataset200_AMOS")
splits_folder = os.path.join(root_dir, "splits")
meta_data_path = os.path.join(root_dir, "labeled_data_meta_0000_0599.csv")
input_images_folder = os.path.join(input_folder, "imagesTr")
input_labels_folder = os.path.join(input_folder, "labelsTr")


def saveDatasetInfo():
    # open metadata
    df = pd.read_csv(meta_data_path)

    amos_id = df["amos_id"].values
    sex_mf = df["Patient's Sex"].values
    genders = np.zeros(sex_mf.shape)
    genders[sex_mf == "F"] = 1
    patients = []

    # Change the patient ID to be a 4-character string with zero padding where necessary
    for id in amos_id:
        id = str(id)
        if len(id) == 1:
            id = "000" + id
        elif len(id) == 2:
            id = "00" + id
        elif len(id) == 3:
            id = "0" + id

        patients.append(id)

    patients = np.array(patients)

    # Save lists
    info = {"patients": patients,
            "genders": genders}

    f = open(os.path.join(input_folder, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()


def main():
    saveDatasetInfo()



if __name__ == "__main__":
    main()
