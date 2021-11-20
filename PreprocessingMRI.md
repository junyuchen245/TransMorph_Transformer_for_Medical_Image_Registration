1. Install FreeSurfer from https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
2. ```export FREESURFER_HOME=/your_freesurfer_directory```
3. ```source $FREESURFER_HOME/SetUpFreeSurfer.sh```
4. ```export SUBJECTS_DIR=/dataset_directory```
5. ```recon-all -parallel -i dataset_directory/img_name.nii -autorecon1 -subjid img_name``` -> This step does motion correction, skull stripping, affine transform comuptation, and intensity normalization.
6. ```mri_convert dataset_directory/img_name/mri/brainmask.mgz  dataset_directory/img_name/mri/brainmask.nii.gz``` -> This step converts the preprocessed image from .mgz into .nii format.
7. ```mri_convert  dataset_directory/img_name/mri/brainmask.mgz --apply_transform dataset_directory/img_name/mri/transforms/talairach.xfm -o dataset_directory/img_name/mri/brainmask_align.mgz``` -> This step does affine tranform to Talairach space.
8. ```mri_convert dataset_directory/img_name/mri/brainmask_align.mgz  dataset_directory/img_name/mri/brainmask_align.nii.gz``` -> This step converts the transformed image from .mgz into .nii format.
9. ```recon-all -parallel -s dataset_directory/img_name.nii -subcortseg -subjid img_name``` -> This step does subcortical segmentation.
10. ```mri_convert dataset_directory/img_name/mri/aseg.auto.mgz  dataset_directory/img_name/mri/aseg.nii.gz``` -> This step converts label image from .mgz into .nii format.
11. ```mri_convert -rt nearest dataset_directory/img_name/mri/aseg.auto.mgz --apply_transform dataset_directory/img_name/mri/transforms/talairach.xfm -o dataset_directory/img_name/mri/aseg_align.mgz``` -> This step does affine tranform to Talairach space using nearest neighbor interpolation for label image.
12. ```mri_convert dataset_directory/img_name/mri/aseg_align.mgz  dataset_directory/img_name/mri/aseg_align.nii.gz``` -> This step converts the transformed label image from .mgz into .nii format.

Note that these steps may take up to **12-24 hours per image** base on our experience. Therefore running these commands in parallel on a server or a cluster is recommended.
