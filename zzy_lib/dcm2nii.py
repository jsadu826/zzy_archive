import os
import subprocess
from multiprocessing import Pool

import SimpleITK as sitk
from tqdm import tqdm


def dcm2nii(args):
    dcm_dir, nii_path = args

    # ==========================================================================
    cmd = [
        "/path/to/dcm2nix/executable",  # NOTE - modify path to dcm2nii executable as needed
        "-9",  # highest compression level
        "-b",  # auxiliary files?
        "n",  # none
        "-d",  # dcm search depth
        "0",  # set to 0 to ignore subdirs
        "-z",  # gzip?
        "y",  # yes, will automatically append .nii.gz to filename
        "-o",
        os.path.dirname(nii_path),  # output dir
        "-f",
        os.path.basename(nii_path).replace(".nii.gz", ""),  # file base name
        dcm_dir,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"================ Error converting {dcm_dir}:")
        print(e.stdout.decode("utf-8"))
        print(e.stderr.decode("utf-8"))
    # ==========================================================================

    # ==========================================================================
    # try:
    #     series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir)
    #     series_reader = sitk.ImageSeriesReader()
    #     series_reader.SetFileNames(series_file_names)
    #     image = series_reader.Execute()
    #     sitk.WriteImage(image=image, fileName=nii_path, useCompression=True)
    # except Exception as e:
    #     print(f"Error converting {dcm_dir}: {str(e)}")
    # ==========================================================================


def dcm2nii_mp(dcm_dirs, nii_root, num_processes):
    os.makedirs(nii_root, exist_ok=True)
    args_list = []
    for dcm_dir in dcm_dirs:
        # NOTE - modify nii path as needed
        nii_path = os.path.join(nii_root, f"{os.path.basename(dcm_dir).replace('.','_')}.nii.gz")
        args_list.append((dcm_dir, nii_path))
    if num_processes > 1:
        with Pool(num_processes) as pool:
            for _ in tqdm(pool.imap(dcm2nii, args_list), total=len(args_list)):
                pass
    else:
        for args in tqdm(args_list, total=len(args_list)):
            dcm2nii(args)
