import os
import operator
import numpy as np
import SimpleITK as sitk
import sys
sys.path.append('/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/codes_nnUNet/data-utils')

from data_util import get_arr_from_nrrd, get_bbox, generate_sitk_obj_from_npy_array
#from scipy.ndimage import sobel, generic_gradient_magnitude
from scipy import ndimage
from SimpleITK.extra import GetArrayFromImage
from scipy import ndimage
import cv2
import matplotlib as plt
from scipy.signal import find_peaks

def crop_top(patient_id, img, seg, crop_shape, return_type, output_img_dir, 
             output_seg_dir, image_format):

    """
    Will crop around the center of bbox of label.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    
    # get image, arr, and spacing
    #image_object = sitk.ReadImage(img_dir)
    image_arr = sitk.GetArrayFromImage(img)
    image_origin = img.GetOrigin()
    #label_object = sitk.ReadImage(seg_dir)
    label_arr = sitk.GetArrayFromImage(seg)
    label_origin = seg.GetOrigin()
    #assert image_arr.shape==label_arr.shape, "image & label shape do not match!"
    #print('max seg value:', np.max(label_arr))    
    # get center. considers all blobs
    bbox = get_bbox(label_arr)
    # returns center point of the label array bounding box
    Z, Y, X = int(bbox[9]), int(bbox[10]), int(bbox[11]) 
    #print('Original Centroid: ', X, Y, Z)
    
    #find origin translation from label to image
    #print('image origin: ', image_origin, 'label origin: ', label_origin)
    origin_dif = tuple(np.subtract(label_origin, image_origin).astype(int))
    #print('origin difference: ', origin_dif)
    
    X_shift, Y_shift, Z_shift = tuple(np.add((X, Y, Z), np.divide(origin_dif, (1, 1, 3)).astype(int)))
    #print('Centroid shifted:', X_shift, Y_shift, Z_shift) 
    c, y, x = image_arr.shape
    
    ## Get center of mass to center the crop in Y plane
    mask_arr = np.copy(image_arr) 
    mask_arr[mask_arr > -500] = 1
    mask_arr[mask_arr <= -500] = 0
    mask_arr[mask_arr >= -500] = 1 
    #print('mask_arr min and max:', np.amin(mask_arr), np.amax(mask_arr))
    centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
    cpoint = c - crop_shape[2]//2
    #print('cpoint, ', cpoint)
    centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])   
    #print('center of mass: ', centermass)
    startx = int(centermass[0] - crop_shape[0]//2)
    starty = int(centermass[1] - crop_shape[1]//2)      
    #startx = x//2 - crop_shape[0]//2       
    #starty = y//2 - crop_shape[1]//2
    startz = int(c - crop_shape[2])
    #print('start X, Y, Z: ', startx, starty, startz)
     
    #---cut bottom slices---
    image_arr = image_arr[30:, :, :]
    label_arr = label_arr[30:, :, :]
    
    #-----normalize CT data signals-------
    norm_type = 'np_clip'
    #image_arr[image_arr <= -1024] = -1024
    ## strip skull, skull UHI = ~700
    #image_arr[image_arr > 700] = 0
    ## normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
    if norm_type == 'np_interp':
        image_arr = np.interp(image_arr, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        image_arr = np.clip(image_arr, a_min=-175, a_max=275)
        MAX, MIN = image_arr.max(), image_arr.min()
        image_arr = (image_arr - MIN) / (MAX - MIN)

    # crop and pad array
    if startz < 0:
        image_arr = np.pad(
            image_arr,
            ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 
            'constant', 
            constant_values=-1024)
        label_arr = np.pad(
            label_arr,
            ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 
            'constant', 
            constant_values=0)
        image_arr_crop = image_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
        label_arr_crop = label_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
    else:
        image_arr_crop = image_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
        label_arr_crop = label_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
    
    # save nrrd
    output_img = output_img_dir + '/' + patient_id + '.' + image_format
    output_seg = output_seg_dir + '/' + patient_id + '.' + image_format
    # save image
    img_sitk = sitk.GetImageFromArray(image_arr_crop)
    img_sitk.SetSpacing(img.GetSpacing())
    img_sitk.SetOrigin(img.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_img)
    writer.SetUseCompression(True)
    writer.Execute(img_sitk)
    # save label
    seg_sitk = sitk.GetImageFromArray(label_arr_crop)
    seg_sitk.SetSpacing(seg.GetSpacing())
    seg_sitk.SetOrigin(seg.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_seg)
    writer.SetUseCompression(True)
    writer.Execute(seg_sitk)

from matplotlib import pyplot as plt    
def crop_top_image_only(patient_id, img_dir, crop_shape, return_type, output_dir, image_format):
    """
    Will center the image and crop top of image after it has been registered.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    #Get original Array and info from image
    image_obj = img_dir
    image_arr = sitk.GetArrayFromImage(image_obj)
    image_origin = image_obj.GetOrigin()   
    image_spacing = image_obj.GetSpacing()
    c, y, x = image_arr.shape
 
    #get the mask from bone desity and higher to calculate the x, y image moment
    mask_arr = np.copy(image_arr) 
    mask_arr[mask_arr <= 700] = 0
    mask_arr = np.sum(mask_arr, axis=0)
    #Treshhold before finding moment
    ret,thresh = cv2.threshold(mask_arr, 0, 255, 0)
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    startx = int(cX - crop_shape[0]//2)
    starty = int(cY - crop_shape[1]//2)  
   
    #-----normalize CT data signals-------
    norm_type = 'np_clip'
    if norm_type == 'np_interp':
        image_arr = np.interp(image_arr, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        image_arr = np.clip(image_arr, a_min=-175, a_max=275)
        MAX, MIN = image_arr.max(), image_arr.min()
        image_arr = (image_arr - MIN) / (MAX - MIN)
        c, y, x = image_arr.shape
    
    #Check for skull only, air only and empty slices
    #Becuse of the registeration, image must be readjusted and crop to elimiate any blank 
    #Blank empty voxel is found by looking at standard dev. if std is zero, it mean blank slice 
    xyArea = int(x*y)
    stdZ = [round(np.std(block), 2) for block in image_arr]
    notzeroZ = [int(np.count_nonzero(block==0))/xyArea for block in image_arr]
    notoneZ = [int(np.count_nonzero(block==1))/xyArea for block in image_arr]
    perStdZ = np.percentile(stdZ, 50)
    perNotzeroZ = np.percentile(notzeroZ, 50)
    perNotoneZ = np.percentile(notoneZ, 50)
    checkZ = [i for i, e in enumerate(zip(stdZ, notzeroZ, notoneZ)) if e[0] > perStdZ and e[1] > perNotzeroZ and e[2] > perNotoneZ]
    stdZ_Loc = checkZ[-1]
    for iz in reversed(checkZ):
        if iz < 100:
            stdZ_Loc = iz 
            break
    if stdZ_Loc > crop_shape[2]:
        endz = stdZ_Loc
    else:
        endz = c
    #Crop to the specify dimensions
    startz = int(endz-crop_shape[2])
    image_arr_crop = image_arr[startz:endz, starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
    
    #Save image
    save_dir = output_dir + '/' + patient_id + '.' + image_format
    new_sitk_object = sitk.GetImageFromArray(image_arr_crop)
    new_sitk_object.SetSpacing(image_spacing)
    new_sitk_object.SetOrigin(image_origin)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(save_dir)
    writer.SetUseCompression(True)
    writer.Execute(new_sitk_object)

def crop_upper_body(img_dir, z_BOTTOM, z_TOP):
    #Crop top of the image to certain z indexs
    #Note that the indexes are reverse, bottom index is top of the head and 0 is the bottom of the scan
    img_arr = sitk.GetArrayFromImage(img_dir)
    zz, yy, xx = np.shape(img_arr)
    #Find the wide part of the skull 
    skull_arr = np.copy(img_arr)     
    skull_arr[skull_arr <= 700] = 0
    maxSkull = []
    for iS in skull_arr[:, :, xx//2]:
        i_peaks, _ = find_peaks(iS)
        diff_list = []
        try:
            if i_peaks.any(): 
                for iiS in range(1,len(i_peaks)):
                    diff_list = []
                    xP = i_peaks[iiS] - i_peaks[iiS-1]
                    diff_list.append(xP)
                maxSkull.append(np.max(diff_list))
            else:
                maxSkull.append(0)
        except ValueError:
            maxSkull.append(0)
    wideSkull = np.argmax(maxSkull)
    cutLoc = wideSkull
    if cutLoc >= z_TOP-z_BOTTOM:  
         # Note when trim the bottom is top (i.e. direction reverse)
         image_arr = img_arr[:cutLoc, :, :]
         zz, yy, xx = image_arr.shape
    
    new_z_TOP = zz-z_BOTTOM
    new_z_BOTTOM = new_z_TOP-z_TOP+z_BOTTOM 
    img_arr = img_arr[new_z_BOTTOM:new_z_TOP, :, :]
    new_img = sitk.GetImageFromArray(img_arr)
    new_img.SetSpacing(img_dir.GetSpacing())
    new_img.SetOrigin(img_dir.GetOrigin())
    
    return new_img

