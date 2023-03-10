import sys, os, glob
import SimpleITK as sitk
import pydicom
import numpy as np



def nrrd_reg_rigid(patient_id, moving_img, output_dir, fixed_img, image_format):
    
    """
    Registers two CTs together: effectively registers CT-PET and PET to the CT-sim and saves 3 files + 1 transform
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri..)
        input_dir (str): Path to folder initial nrrd image files
        output_dir (str): Path to folder where the registered nrrds will be saved.
    Returns:
        The sitk image object.
    Raises:
        Exception if an error occurs.
    
    
    for file in os.listdir(input_dir): #LOOP Goes through nrrd raw images etc
        mask_arr = []
        if not file.startswith('.') and 'CT' in file:
            patient_id = file.split('_')[1][0:11]
            #modality = file.split('_')[2]
            print("patient ID: ", patient_id)
            path_image = os.path.join(input_dir,file)            
            print("image path: ",path_image)"""

    #fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32) # PATH FOR FIXED IMAGE
    #moving_image = sitk.ReadImage(img_dir, sitk.sitkFloat32)
    transform = sitk.CenteredTransformInitializer(
        fixed_img, 
        moving_img, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    #multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)
    final_transform = registration_method.Execute(fixed_img, moving_img)                               
    
    reg_img = sitk.Resample(
        moving_img, 
        fixed_img, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_img.GetPixelID())
    
    if output_dir != '':
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_dir + '/' + patient_id + '.' + image_format)
        writer.SetUseCompression(True)
        writer.Execute(reg_img)

    return reg_img, fixed_img, moving_img, final_transform


def bspline_intra_modal_registration(fixed_image, moving_image, fixed_image_mask=None, fixed_points=None, moving_points=None):
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 
    grid_physical_spacing = [50.0, 50.0, 50.0] # A control point every 50mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                         transformDomainMeshSize = mesh_size, order=3)    
    registration_method.SetInitialTransform(initial_transform)
        
    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    
    
    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
    
    return registration_method.Execute(fixed_image, moving_image)



def demons_registration(fixed_image, moving_image, fixed_points = None, moving_points = None):
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))
    
    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0) 
    
    registration_method.SetInitialTransform(initial_transform)
    
    registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU
        
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])    
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)        
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
        
    return registration_method.Execute(fixed_image, moving_image)    




'''
##PERFORM REGISTRATION   
# Select the fixed and moving images, valid entries are in [0,9].

fixed_image = sitk.ReadImage('/Volumes/BK_RESEARCH/HN_PETSEG/curated/CHUM_files/0_image_raw_CHUM_registered/HN-CHUM-004_CT-SIM.nrrd', sitk.sitkFloat32)
moving_image = sitk.ReadImage('/Volumes/BK_RESEARCH/HN_PETSEG/curated/CHUM_files/0_image_raw_CHUM_registered/HN-CHUM-004_CT-PET_registered.nrrd', sitk.sitkFloat32)

#tx = bspline_intra_modal_registration(fixed_image, moving_image)
tx = demons_registration(fixed_image, moving_image)

# Transfer the segmentation via the estimated transformation. Use Nearest Neighbor interpolation to retain the labels.

transformed_labels = sitk.Resample(moving_image,
                                   fixed_image,
                                   tx, 
                                   sitk.sitkNearestNeighbor,
                                   0.0, 
                                   moving_image.GetPixelID())

sitk.WriteImage(transformed_labels, os.path.join(output_dir, "HN-CHUM-004" + "_CT-PET_registered_bspline.nrrd"))

sitk.WriteImage(transformed_labels, os.path.join(output_dir, "HN-CHUM-004" + "_CT-PET_registered_bspline.nrrd"))

moving_pet_resampled = sitk.Resample(moving_pet, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
sitk.WriteImage(moving_pet_resampled, os.path.join(output_dir, patient_id + "_PET_registered_bspline.nrrd"))
sitk.WriteTransform(final_transform, os.path.join(output_dir, patient_id + "_registration_transform.tfm"))


segmentations_before_and_after = [masks[moving_image_index], transformed_labels]
interact(display_coronal_with_label_maps_overlay, coronal_slice = (0, images[0].GetSize()[1]-1),
         mask_index=(0,len(segmentations_before_and_after)-1),
         image = fixed(images[fixed_image_index]), masks = fixed(segmentations_before_and_after), 
         label=fixed(lung_label), window_min = fixed(-1024), window_max=fixed(976));
'''