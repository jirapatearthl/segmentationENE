#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:24:12 2022

@author: likitler
"""

import sys
import os
import pydicom
import glob
import SimpleITK as sitk
import pandas as pd
import numpy as np
from interpolate import interpolate
from registration import nrrd_reg_rigid
import SimpleITK as sitk
import shutil

import json
import glob
from pathlib import Path

import re

from crop_image_18Dec2022 import crop_top, crop_top_image_only, crop_upper_body#crop_full_body

root_dir = '/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/data4Seg'#'/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
folder_in = 'RAW_folder'
folder_out = 'input_folder_XL'
image_format = 'nrrd'

#crop_shape = (160, 160, 64)
crop_shape = (192, 192, 96)

#location of template for registeration 
fixed_img_dir = '/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/codes_nnUNet/10020741814.nrrd'
fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)

path_in = os.path.join(root_dir, folder_in)
path_out = os.path.join(root_dir, folder_out)

img_nrrd_ids = []
img_nii_ids = []

#Find the raw data and get ready to convert to unnet file name format
list_in = glob.glob(path_in + '/*')
for i in list_in:
    listNrrd = glob.glob(i + '/*.nrrd') 
    datasetNow = os.path.basename(i)
    datasetNow = datasetNow.split('_')[0]
    datasetPath_out = os.path.join(path_out, datasetNow)
    Path(datasetPath_out).mkdir(parents=True, exist_ok=True)
    #convert to unnet file name format
    for ii in listNrrd:
        patNow = os.path.basename(ii)
        filePath_out = datasetPath_out 
        patNow = patNow[:-5]
        patNii = patNow+'.nii.gz'        
       
        datasetNow = patNow.split('_')[0]
        if  datasetNow == 'E3311':
            patIdNow = patNow.split('_')[2]
            patIdNow = [int(s) for s in re.findall(r'\d+', patIdNow)][0]
            finalIdNow = str(patIdNow).zfill(8)
        elif datasetNow == 'DFCI19009':
            patIdNow = patNow.split('_')[1]
            patIdNow = [int(s) for s in re.findall(r'\d+', patIdNow)][0]
            finalIdNow = str(patIdNow).zfill(8)
            datasetNow = 'BWH'
        elif datasetNow == 'HMS':
            patIdNow = patNow.split('_')[1]
            patIdNow = [int(s) for s in re.findall(r'\d+', patIdNow)][0]
            finalIdNow = str(patIdNow).zfill(8)
            datasetNow = 'BWH'
        else: 
            patIdNow = patNow.split('_')[1]
            patIdNow = [int(s) for s in re.findall(r'\d+', patIdNow)][0]
            finalIdNow = str(patIdNow).zfill(8)
        
        finalNameNow = datasetNow + '_' + finalIdNow + '_' + str('').zfill(4)
        
        img_dir = ii
        img_id = finalNameNow
        print(img_id)
 
        #interplolate
        try: 
            img_interp = interpolate( 
                patient_id=img_id,
                path_to_nrrd=img_dir, 
                interpolation_type='linear', #"linear" for image
                new_spacing=(1, 1, 3), 
                return_type='sitk_obj',
                output_dir='',
                image_format=image_format)
        #Call funtion that limited the top half of the body depending on the value you set
            xx,yy,zz = img_interp.GetSize()
            if zz > 200:
                new_zz = 200
                img_interp = crop_upper_body(img_interp, 0, new_zz)
        #register current participant to the template 
            reg_img, fixed_img, moving_img, final_transform = nrrd_reg_rigid( 
                patient_id=img_id, 
                moving_img=img_interp, 
                output_dir='', 
                fixed_img=fixed_img, 
                image_format=image_format)
        #crop top image only
            img_RegCrop_dir = datasetPath_out
            crop_top_image_only(patient_id=img_id, 
                img_dir=reg_img,  
                crop_shape=crop_shape, 
                return_type='sitk_object', 
                output_dir=img_RegCrop_dir,
                image_format='nii.gz')
            
        except Exception as e:
            print(e, 'crop failed!')
    #except Exception as e:
    #    print("Fail")
        
        #saveOutputNow = os.path.join(datasetPath_out, finalNameNow+'.nii.gz')
        #sitk.WriteImage(imgSitkNow, saveOutputNow)
        #img_nrrd_ids.append(patNow)
        #img_nii_ids.append(finalNameNow)
