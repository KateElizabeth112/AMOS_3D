# explore the AMOS metadata
import pandas as pd
import numpy as np
import os
import pickle as pkl
import shutil

local = False

if local:
    data_dir = "/Users/katecevora/Documents/PhD/data/AMOS_3D"
    meta_data_path = os.path.join(data_dir, "labeled_data_meta_0000_0599.csv")
    splits_folder = os.path.join(data_dir, "splits")
else:
    input_folder = "/vol/biomedic3/kc2322/data/AMOS_3D/nnUNet_raw/Dataset200_AMOS"
    meta_data_path = os.path.join("/vol/biomedic3/kc2322/data/AMOS_3D", "labeled_data_meta_0000_0599.csv")
    input_images_folder = os.path.join(input_folder, "imagesTr")
    input_labels_folder = os.path.join(input_folder, "labelsTr")

    output_folder = "/vol/biomedic3/kc2322/data/AMOS_3D/nnUNet_raw/Dataset200_AMOS"
    splits_folder = os.path.join("/vol/biomedic3/kc2322/data/AMOS_3D", "splits")


def list_train_images():
    images = os.listdir(input_images_folder)
    ids = []
    for img in images:
        if img.endswith(".nii.gz"):
            id = img[5:9]
            print(id)
            ids.append(int(id.lstrip('0')))

    # save
    f = open(os.path.join(output_folder, "tr_ids.pkl"), "wb")
    pkl.dump(ids, f)
    f.close()


def generate_sets(meta_path):
    # Extract relevant meta data and create numpy arrays of male and female IDs
    meta = pd.read_csv(meta_path)
    ids_m_orig = np.array(meta[meta["Patient's Sex"] == "M"]["amos_id"].values)
    ids_f_orig = np.array(meta[meta["Patient's Sex"] == "F"]["amos_id"].values)

    # keep only the ids that have both an image and label (original training set)
    f = open(os.path.join(output_folder, "tr_ids.pkl"), "rb")
    ids = pkl.load(f)
    f.close()

    ids_m = []
    ids_f = []

    for id in list(ids):
        if id in ids_m_orig:
            # add to list of useable ids
            ids_m.append(id)
        else:
            # remove from list of useable ids
            ids_f.append(id)

    ids_m = np.array(ids_m)
    ids_f = np.array(ids_f)

    # Figure out our training and test set sizes
    num_m = ids_m.shape[0]
    num_f = ids_f.shape[0]

    print("Number of males: {}".format(num_m))
    print("Number of females: {}".format(num_f))

    ts_size = 40
    tr_size = int(num_f) - int(ts_size / 2)

    print("Training set size: {}".format(tr_size))
    print("Test set size: {}".format(ts_size))

    ids_tr_m = ids_m[:tr_size]
    ids_ts_m = ids_m[tr_size:int(tr_size + ts_size / 2)]
    ids_tr_f = ids_f[:tr_size]
    ids_ts_f = ids_f[tr_size:]

    # randomly shuffle indices
    np.random.shuffle(ids_m)
    np.random.shuffle(ids_f)

    ids_ts = np.concatenate((ids_ts_f, ids_ts_m), axis=0)

    # Set 1 train: 225 men, 225 women
    # Set 1 test: 49 men, 49 women
    ids_tr = np.concatenate((ids_tr_f[:int(tr_size / 2)], ids_tr_m[:int(tr_size / 2)]), axis=0)

    set_1_ids = {"train": ids_tr, "test": ids_ts}
    f = open(os.path.join(splits_folder, "set1_splits.pkl"), "wb")
    pkl.dump(set_1_ids, f)
    f.close()

    # Set 2 train: 0 men, 450 women
    # Set 2 test: 49 men, 49 women
    set_2_ids = {"train": ids_tr_f, "test": ids_ts}
    f = open(os.path.join(splits_folder, "set2_splits.pkl"), "wb")
    pkl.dump(set_2_ids, f)
    f.close()

    # Set 3 train: 450 men, 0 women
    # Set 3 test: 49 men, 49 women
    set_3_ids = {"train": ids_tr_m, "test": ids_ts}
    f = open(os.path.join(splits_folder, "set3_splits.pkl"), "wb")
    pkl.dump(set_3_ids, f)
    f.close()


def copy_images(dataset_name, ids_tr, ids_ts):
    os.mkdir(os.path.join(output_folder, dataset_name))

    output_imagesTr = os.path.join(output_folder, dataset_name, "imagesTr")
    output_labelsTr = os.path.join(output_folder, dataset_name, "labelsTr")
    output_imagesTs = os.path.join(output_folder, dataset_name, "imagesTs")
    output_labelsTs = os.path.join(output_folder, dataset_name, "labelsTs")

    os.mkdir(output_imagesTr)
    os.mkdir(output_labelsTr)
    os.mkdir(output_imagesTs)
    os.mkdir(output_labelsTs)

    # copy over the files from Training Set
    for case in list(ids_tr):
        # pad the id if necessary
        id = str(case)
        id_len = len(id)
        if id_len < 4:
            for j in range(4 - id_len):
                id = "0" + id

        print("Case {}".format(id))

        img_name = "amos_" + id + "_0000.nii.gz"
        lab_name = "amos_" + id + ".nii.gz"
        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTr, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTr, lab_name))

    # copy over the files from Test Set
    for case in list(ids_ts):
        # pad the id if necessary
        id = str(case)
        id_len = len(id)
        if id_len < 4:
            for j in range(4 - id_len):
                id = "0" + id

        print("Case {}".format(id))

        img_name = "amos_" + id + "_0000.nii.gz"
        lab_name = "amos_" + id + ".nii.gz"

        # Copy across images
        shutil.copyfile(os.path.join(input_images_folder, img_name), os.path.join(output_imagesTs, img_name))

        # Copy across labels
        shutil.copyfile(os.path.join(input_labels_folder, lab_name), os.path.join(output_labelsTs, lab_name))


def sort_sets():
    # Sort the case IDs according to the sets
    # Set1
    f = open(os.path.join(splits_folder, "set1_splits.pkl"), "rb")
    set_1_ids = pkl.load(f)
    f.close()

    ids_tr = set_1_ids["train"]
    ids_ts = set_1_ids["test"]

    print("Working on Set 1....")
    copy_images("Dataset701_Set1", ids_tr, ids_ts)

    # Set2
    f = open(os.path.join(splits_folder, "set2_splits.pkl"), "rb")
    set_2_ids = pkl.load(f)
    f.close()

    ids_tr = set_2_ids["train"]
    ids_ts = set_2_ids["test"]

    print("Working on Set 1....")
    copy_images("Dataset702_Set2", ids_tr, ids_ts)

    # Set3
    f = open(os.path.join(splits_folder, "set3_splits.pkl"), "rb")
    set_3_ids = pkl.load(f)
    f.close()

    ids_tr = set_3_ids["train"]
    ids_ts = set_3_ids["test"]

    print("Working on Set 1....")
    copy_images("Dataset703_Set3", ids_tr, ids_ts)


def main():
    #list_train_images()
    generate_sets(meta_data_path)
    sort_sets()


if __name__ == "__main__":
    main()
