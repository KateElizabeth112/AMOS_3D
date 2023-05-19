# Slice 3D nii images to 2D
import numpy as np
import os
import nibabel as nib

#root_dir = '/Users/katecevora/Documents/PhD'
root_dir = 'vol/biomedic3/kc2322/'

img_dir_in = os.path.join(root_dir, 'data/AMOS_3D/nnUNet_raw/Dataset200_AMOS/imagesVa')
lab_dir_in = os.path.join(root_dir, 'data/AMOS_3D/nnUNet_raw/Dataset200_AMOS/labelsVa')

img_dir_out = os.path.join(root_dir, 'data/AMOS_3D/nnUNet_raw/Dataset200_AMOS/imagesVaSorted')
lab_dir_out = os.path.join(root_dir, 'data/AMOS_3D/nnUNet_raw/Dataset200_AMOS/labelsVaSorted')


def prep2DData():
    files = os.listdir(img_dir_in)

    for f in files:
        if f.endswith(".nii.gz"):
            try:
                print(f)

                # open label and image (shape is H, W, D)
                img_nii = nib.load(os.path.join(img_dir_in, f))
                gt_nii = nib.load(os.path.join(lab_dir_in, f))

                if img_nii.shape[0] != 512:
                    raise Exception("Wrong size, ignoring")

                if img_nii.shape[1] != 512:
                    raise Exception("Wrong size, ignoring")

                nib.save(gt_nii, os.path.join(lab_dir_out, f.split('.')[0] + ".nii.gz"))
                nib.save(img_nii, os.path.join(img_dir_out, f.split('.')[0] + "_0000.nii.gz"))
            except:
                print(f + " failed")
                continue


def main():
    prep2DData()


if __name__ == "__main__":
    main()
