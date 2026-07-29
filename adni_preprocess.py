# adni_preprocess.py
"""
Full MRI preprocessing pipeline for ADNI T1-weighted images:
1. Skull-stripping (HD-BET)
2. Bias field correction (N4)
3. Registration to MNI152 template (ANTs)
4. Resampling to fixed voxel grid
5. Intensity normalization (z-score)

Outputs preprocessed NIfTI files in the specified output folder.
"""

import os
from pathlib import Path
import ants
import nibabel as nib
import numpy as np
from subprocess import run
from tqdm import tqdm
from typing import Tuple
import SimpleITK as sitk
from antspynet.utilities import brain_extraction

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = "adni_raw_nifti"       # folder containing raw NIfTI files
OUTPUT_DIR = "adni_preprocessed"   # folder to save processed NIfTIs
MNI_TEMPLATE = "MNI152_T1_1mm.nii.gz"  # path to MNI template
TARGET_SHAPE = (182, 218, 182)     # resample shape (can adjust)
DEVICE = "cpu"                     # 'cuda' if GPU available



import nibabel as nib
import tensorflow as tf
import tensorflow.compat.v1 as tf1

# Patch the tensorflow module so deepbrain sees v1 attributes
tf.Session = tf1.Session
tf.gfile = tf1.gfile
tf.GraphDef = tf1.GraphDef

tf1.disable_v2_behavior()

from deepbrain import Extractor

extractor = Extractor()

def normalize_zscore(img):
    # Standardizes intensities: mean=0, variance=1
    return sitk.Normalize(img)

def resize_image(img, new_size=(128, 128, 128)):
    # Calculate new spacing to maintain physical size
    reference_size = new_size
    reference_spacing = [
        orig_sz * orig_spc / new_sz 
        for orig_sz, orig_spc, new_sz in zip(img.GetSize(), img.GetSpacing(), reference_size)
    ]
    
    # Create a reference "empty" image with the new dimensions
    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing(reference_spacing)
    
    # Resample
    return sitk.Resample(img, reference_image, sitk.Transform(), sitk.sitkLinear)



def registration(moving_sitk_img, mni_template_path):
    fixed_img = sitk.ReadImage(str(mni_template_path), sitk.sitkFloat32)
    moving_img = sitk.Cast(moving_sitk_img, sitk.sitkFloat32)

    # 1. Align centers
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1)
    
    # IMPORTANT: Prevent distortion by scaling parameters properly
    R.SetOptimizerScalesFromPhysicalShift() 
    R.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100)
    
    # 2. Multi-resolution approach
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])

    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)

    final_transform = R.Execute(fixed_img, moving_img)
    
    return sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())

def skull_strip(sitk_image, ext=extractor):
    # why: deepbrain needs a numpy array, not a SimpleITK object.
    # We convert the piped object to numpy.
    # data = sitk.GetArrayFromImage(sitk_image)
    
    # # output: A 3D probability map (0 to 1) of "how likely this is brain tissue".
    # prob = ext.run(data) 
    
    # # why: 0.5 is the standard threshold. > 0.5 becomes 1 (brain), <= 0.5 becomes 0 (background).
    # mask = prob > 0.5
    # brain_data = data * mask
    data = sitk.GetArrayFromImage(sitk_image)
    data_transposed = data.transpose(2, 1, 0)  # sitk -> deepbrain orientation

    prob = ext.run(data_transposed)
    mask = prob > 0.3

    # Transpose back before reconstructing
    brain_data = data * mask.transpose(2, 1, 0)
    
    # TO PIPE: We convert back to SimpleITK to preserve spatial metadata for registration.
    brain_img = sitk.GetImageFromArray(brain_data)
    brain_img.CopyInformation(sitk_image)
    
    return brain_img



def n4_bias_field_correction(
    raw_img_path: str,
    shrink_factor: int = 4,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Apply N4 bias field correction to a medical image using SimpleITK.

    This function performs the following steps:
    1. Reads the image from disk and enforces RPS orientation.
    2. Rescales intensity to [0, 255].
    3. Computes a foreground (head) mask using Li thresholding.
    4. Shrinks the image and mask for faster bias field estimation.
    5. Estimates the bias field using the N4 algorithm.
    6. Applies the estimated bias field to the full-resolution image.

    Parameters
    ----------
    raw_img_path : str
        Path to the input medical image (e.g., NIfTI, DICOM).
    shrink_factor : int, optional (default=4)
        Factor by which to downsample the image for faster bias field estimation.
        Higher values increase speed but may reduce accuracy.

    Returns
    -------
    corrected_full_res : sitk.Image
        Bias-corrected image at full resolution.
    head_mask : sitk.Image
        Binary mask used during bias field estimation.

    Notes
    -----
    - Uses Li thresholding for mask generation (robust for many MRI modalities).
    - N4 correction is computationally expensive; shrinking speeds up estimation.
    - The correction is applied back to the original resolution image.
    """

    # ------------------------------------------------------------------
    # Step 1: Load image and standardize orientation
    # ------------------------------------------------------------------
    raw_img = sitk.ReadImage(raw_img_path, sitk.sitkFloat32)
    raw_img = sitk.DICOMOrient(raw_img, "RPS")  # Standard orientation

    # ------------------------------------------------------------------
    # Step 2: Intensity normalization (rescale to [0, 255])
    # ------------------------------------------------------------------
    rescaled_img = sitk.RescaleIntensity(raw_img, 0, 255)

    # ------------------------------------------------------------------
    # Step 3: Generate foreground mask using Li thresholding
    # ------------------------------------------------------------------
    head_mask = sitk.LiThreshold(rescaled_img, 0,1)

    # ------------------------------------------------------------------
    # Step 4: Shrink image and mask to speed up N4 estimation
    # ------------------------------------------------------------------
    dimension = raw_img.GetDimension()
    shrink_factors = [shrink_factor] * dimension

    input_image_shrunk = sitk.Shrink(raw_img, shrink_factors)
    mask_image_shrunk = sitk.Shrink(head_mask, shrink_factors)

    # ------------------------------------------------------------------
    # Step 5: Estimate bias field using N4 algorithm
    # ------------------------------------------------------------------
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    _ = bias_corrector.Execute(input_image_shrunk, mask_image_shrunk)

    # ------------------------------------------------------------------
    # Step 6: Reconstruct full-resolution bias field and correct image
    # ------------------------------------------------------------------
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img)
    corrected_full_res = raw_img / sitk.Exp(log_bias_field)

    return corrected_full_res, head_mask


if __name__ == "__main__":
    pass