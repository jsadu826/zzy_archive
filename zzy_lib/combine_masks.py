import argparse
import os

import nibabel as nib
import numpy as np


def combine_masks(input_masks_dir, output_path, target_mask_names, suffix):
    print(f"=============== Processing {input_masks_dir}")

    try:
        if target_mask_names is None:
            target_mask_names = []
            for name in sorted(os.listdir(input_masks_dir)):
                if name.endswith(suffix):
                    target_mask_names.append(name)
        else:
            target_mask_names = [name if name.endswith(suffix) else name + suffix for name in target_mask_names]

        class_map = {name: i + 1 for i, name in enumerate(target_mask_names)}

        ref_img = None
        combined_arr = None

        for name in target_mask_names:
            input_path = os.path.join(input_masks_dir, name)
            img = nib.load(input_path)
            arr = img.get_fdata()
            if ref_img is None:
                ref_img = img
                combined_arr = arr
            else:
                combined_arr += arr * class_map[name]

        combined_arr = combined_arr.astype(np.uint8)
        combined_img = nib.Nifti1Image(combined_arr, affine=ref_img.affine, header=ref_img.header)

        if not output_path.endswith(suffix):
            output_path += suffix
        nib.save(combined_img, output_path)

        print(f"Combined mask saved to: {output_path}")
        print(f"Class mapping: {class_map}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_masks_dir", type=str, required=True)
    parser.add_argument("--output_path", default=None, type=str, required=True, help="Can be with or without [suffix].")
    parser.add_argument("--target_mask_names", default=None, type=str, nargs="+", required=False, help="Can be with or without [suffix]. If not provided, will use all files with [suffix] in [input_masks_dir].")
    parser.add_argument("--suffix", default=".nii.gz", type=str, required=False, help="Suffix to identify and save files.")
    args = parser.parse_args()
    combine_masks(args.input_masks_dir, args.output_path, args.target_mask_names, args.suffix)
