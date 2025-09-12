import nibabel as nib
import numpy as np


def copy_affine_from_reference(src_path, ref_path, tgt_path):
    """
    Copy the affine matrix from a reference NIfTI and apply it to the source NIfTI,
    without changing the voxel data.
    """

    src_nii = nib.load(src_path)
    ref_nii = nib.load(ref_path)

    # Preserve dtype
    src_data = src_nii.get_fdata()
    src_dtype = src_nii.get_data_dtype()
    if np.issubdtype(src_dtype, np.integer):
        src_data = np.rint(src_data).astype(src_dtype)
    else:
        src_data = src_data.astype(src_dtype)

    # Save
    tgt_nii = nib.Nifti1Image(src_data, affine=ref_nii.affine, header=src_nii.header)
    tgt_nii.set_sform(ref_nii.affine, code=1)
    tgt_nii.set_qform(ref_nii.affine, code=1)
    nib.save(tgt_nii, tgt_path)
    print(f"✅ copy_affine_from_reference, saved to: {tgt_path}")


def reorient_to_reference(src_path, ref_path, tgt_path):
    """
    Reorient a source NIfTI to match the reference NIfTI's orientation.
    """

    src_nii = nib.load(src_path)
    ref_nii = nib.load(ref_path)

    # Compute orientation transformation
    transform_orient = nib.orientations.ornt_transform(nib.orientations.io_orientation(src_nii.affine), nib.orientations.io_orientation(ref_nii.affine))

    # Apply transformation
    src_data = src_nii.get_fdata()
    tgt_data = nib.orientations.apply_orientation(src_data, transform_orient)

    # Preserve dtype
    src_dtype = src_nii.get_data_dtype()
    if np.issubdtype(src_dtype, np.integer):
        tgt_data = np.rint(tgt_data).astype(src_dtype)
    else:
        tgt_data = tgt_data.astype(src_dtype)

    # Correct affine
    tgt_affine = src_nii.affine @ nib.orientations.inv_ornt_aff(transform_orient, src_nii.shape)

    # Save
    tgt_nii = nib.Nifti1Image(tgt_data, tgt_affine, header=src_nii.header)
    tgt_nii.set_sform(tgt_affine, code=1)
    tgt_nii.set_qform(tgt_affine, code=1)
    nib.save(tgt_nii, tgt_path)
    print(f"✅ reorient_to_reference, saved to: {tgt_path}")


def convert_to_ras(src_path, tgt_path):
    """
    Convert a NIfTI to RAS orientation.
    """

    src_nii = nib.load(src_path)

    # Compute orientation transformation
    transform_orient = nib.orientations.ornt_transform(nib.orientations.io_orientation(src_nii.affine), nib.orientations.axcodes2ornt(("R", "A", "S")))

    # Apply transformation
    src_data = src_nii.get_fdata()
    tgt_data = nib.orientations.apply_orientation(src_data, transform_orient)

    # Preserve dtype
    src_dtype = src_nii.get_data_dtype()
    if np.issubdtype(src_dtype, np.integer):
        tgt_data = np.rint(tgt_data).astype(src_dtype)
    else:
        tgt_data = tgt_data.astype(src_dtype)

    # Correct affine
    tgt_affine = src_nii.affine @ nib.orientations.inv_ornt_aff(transform_orient, src_nii.shape)

    # Save
    tgt_nii = nib.Nifti1Image(tgt_data, tgt_affine, header=src_nii.header)
    tgt_nii.set_sform(tgt_affine, code=1)
    tgt_nii.set_qform(tgt_affine, code=1)
    nib.save(tgt_nii, tgt_path)
    print(f"✅ convert_to_ras, saved to: {tgt_path}")
