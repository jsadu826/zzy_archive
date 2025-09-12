import argparse
import os
from multiprocessing import Pool, cpu_count

import nibabel as nib
import numpy as np
from tqdm import tqdm


def fix_affine(filepath):
    try:
        img = nib.load(filepath)
        new_img = nib.Nifti1Image(
            dataobj=img.get_fdata(),
            affine=np.eye(4),
            header=img.header,
        )
        if img.get_qform(coded=False) is not None:
            new_img.set_qform(img.get_qform(coded=False), update_affine=True)
        elif img.get_sform(coded=False) is not None:
            new_img.set_sform(img.get_sform(coded=False), update_affine=True)
        nib.save(new_img, filepath)
        return True
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        return False


def find_nii_files(folder):
    nii_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".nii.gz") or f.endswith(".nii"):
                nii_files.append(os.path.join(root, f))
    return nii_files


def process_folder(folder, num_workers=4):
    nii_files = find_nii_files(folder)
    if not nii_files:
        print("No .nii(.gz) files found.")
        return
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(fix_affine, nii_files), total=len(nii_files), desc="Processing NIfTI files"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix affine matrices in NIfTI files.")
    parser.add_argument("folder", type=str, help="Path to the folder containing NIfTI files.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes to use.")
    args = parser.parse_args()
    process_folder(args.folder, args.num_workers)
