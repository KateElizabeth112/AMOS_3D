# explore the AMOS metadata
import pandas as pd
import numpy as np
import os
import pickle as pkl
import shutil

local = False

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

    # List the images in the training dataset
    images = os.listdir(input_images_folder)

    amos_id = df["amos_id"].values
    sex_mf = df["Patient's Sex"].values
    scanner = df["Manufacturer's Model Name"].values
    site = df["Site"].values

    patients = []
    genders = []
    patients_tr = []
    scanner_tr = []
    site_tr = []

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

    # cycle over images in the training set folder and extract relevant metadata
    for image in images:
        if image.endswith(".nii.gz"):
            # Find the gender of the subject from the metadata
            id = image[5:9]
            x = np.where(patients == id)[0][0]
            patients_tr.append(id)
            if sex_mf[x] == "M":
                genders.append(0)
            elif sex_mf[x] == "F":
                genders.append(1)

            scanner_tr = scanner[x]
            site_tr = site[x]

    genders = np.array(genders)
    patients_tr = np.array(patients_tr)
    scanner_tr = np.array(scanner_tr)
    site_tr = np.array(site_tr)

    # Save lists
    info = {"patients": patients_tr,
            "genders": genders,
            "scanner": scanner_tr,
            "site": site_tr}

    f = open(os.path.join(input_folder, "info.pkl"), "wb")
    pkl.dump(info, f)
    f.close()


def main():
    saveDatasetInfo()



if __name__ == "__main__":
    main()
