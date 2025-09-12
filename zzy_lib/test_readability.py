import os
import argparse
import SimpleITK as sitk
from multiprocessing import Pool

def get_files(root_dir, exclude_suffixes=None):
    """
    Recursively get all files under root_dir excluding files with given suffixes.
    
    Args:
        root_dir (str): Root directory to search.
        exclude_suffixes (list of str): List of suffixes to exclude, e.g. ['.txt', '.csv'].
        
    Returns:
        List of file paths.
    """
    if exclude_suffixes is None:
        exclude_suffixes = []
        
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if not any(f.endswith(suffix) for suffix in exclude_suffixes):
                all_files.append(os.path.join(dirpath, f))
    return all_files

def check_read_image(file_path):
    """
    Try reading a file using SimpleITK. Return tuple (file_path, success, error_msg)
    """
    try:
        img = sitk.ReadImage(file_path)
        return (file_path, True, None)
    except Exception as e:
        return (file_path, False, str(e))

def main(root_dir, exclude_suffixes=None, num_processes=None):
    # 1. Get all files recursively
    files = get_files(root_dir, exclude_suffixes)
    print(f"Found {len(files)} files to check.")

    # 2. Process files in parallel
    import tqdm
    results = []
    with Pool(processes=num_processes) as pool:
        for res in tqdm.tqdm(pool.imap(check_read_image, files), total=len(files)):
            results.append(res)

    # 3. Summarize results
    success_files = [f for f, success, _ in results if success]
    failed_files = [(f, err) for f, success, err in results if not success]

    print(f"\n✅ Successfully read {len(success_files)} files.")
    print(f"❌ Failed to read {len(failed_files)} files.")
    if failed_files:
        print("\nFailed files:")
        for f, err in failed_files:
            print(f"{f} --> {err}")

    return success_files, failed_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if files can be read using SimpleITK")
    parser.add_argument("root_dir", help="Root directory to search for files")
    parser.add_argument("--exclude-suffixes", nargs="*", default=[], 
                       help="File suffixes to exclude (e.g., .txt .csv)")
    parser.add_argument("--num-processes", type=int, default=8,
                       help="Number of processes to use (default: CPU count - 1)")
    
    args = parser.parse_args()
    
    success_files, failed_files = main(args.root_dir, args.exclude_suffixes, args.num_processes)
