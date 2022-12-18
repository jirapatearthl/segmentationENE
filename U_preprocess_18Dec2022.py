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

##sys.path.append('/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/codes_nnUNet/data-utils')

#from dcm_to_nrrd import dcm_to_nrrd
#from rtstruct_to_nrrd import rtstruct_to_nrrd
#from combine_structures import combine_structures
from interpolate import interpolate
##from crop_image import crop_top, crop_top_image_only, crop_full_body
from registration import nrrd_reg_rigid
import SimpleITK as sitk
import shutil
##import nibabel as ni

import json
import glob
from pathlib import Path

import re

from crop_image_JL import crop_top, crop_top_image_only, crop_upper_body#crop_full_body

root_dir = '/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/data4Seg'#'/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
folder_in = 'RAW_folder'
folder_out = 'input_folder_XL'
'''
MASTER_jsonPATH = '/mnt/InternalHDD/User/likitler/ENE_Project/CODE_Apr2022/JSON/ALLRUN.json'
with open(MASTER_jsonPATH) as f:
  dataJ = json.load(f)
dataTrain = dataJ["TRAIN"]
#dataSupport = dataTrain["inputSupportFilePath"]
dataHeader = dataJ["HEADER"]
dataPre = dataJ["PRE"]
##sys.path.append('/home/bhkann/git-repositories/working-hn-dl-folder/nrrd_model')
#sys.path.append('/mnt/InternalHDD/User/likitler/ENE_Project/CODE_Sep2021/working-hn-dl-folder-master/nrrd_model')
'''

image_format = 'nrrd'
##crop_shape = (160, 160, 64)
crop_shape = (192, 192, 96)
fixed_img_dir = '/mnt/InternalHDD/User/likitler/ENE_Project/Segmentation/HeadNeck/codes_nnUNet/10020741814.nrrd'
fixed_img = sitk.ReadImage(fixed_img_dir, sitk.sitkFloat32)
                           
path_in = os.path.join(root_dir, folder_in)
path_out = os.path.join(root_dir, folder_out)

img_nrrd_ids = []
img_nii_ids = []

list_in = glob.glob(path_in + '/*')
for i in list_in:
    listNrrd = glob.glob(i + '/*.nrrd') 
    datasetNow = os.path.basename(i)
    datasetNow = datasetNow.split('_')[0]
    datasetPath_out = os.path.join(path_out, datasetNow)
    Path(datasetPath_out).mkdir(parents=True, exist_ok=True)
    #img_crop_dir = os.path.join(datasetPath_out, 'RegImg')
    #seg_crop_dir = os.path.join(datasetPath_out, 'RegImg')
    #Path(img_crop_dir).mkdir(parents=True, exist_ok=True)
    #Path(seg_crop_dir).mkdir(parents=True, exist_ok=True)
    
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
            #print(finalIdNow)
        elif datasetNow == 'DFCI19009':
            patIdNow = patNow.split('_')[1]
            patIdNow = [int(s) for s in re.findall(r'\d+', patIdNow)][0]
            finalIdNow = str(patIdNow).zfill(8)
            datasetNow = 'BWH'
            #print(finalIdNow)
        elif datasetNow == 'HMS':
            patIdNow = patNow.split('_')[1]
            patIdNow = [int(s) for s in re.findall(r'\d+', patIdNow)][0]
            finalIdNow = str(patIdNow).zfill(8)
            datasetNow = 'BWH'
            #print(finalIdNow)
        else: 
            patIdNow = patNow.split('_')[1]
            patIdNow = [int(s) for s in re.findall(r'\d+', patIdNow)][0]
            finalIdNow = str(patIdNow).zfill(8)
            #print(finalIdNow)
        
        finalNameNow = datasetNow + '_' + finalIdNow + '_' + str('').zfill(4)
        print(patNow)
        print(finalNameNow)
        
        ##imgSitkNow = sitk.ReadImage(ii)
        ##z_img = imgSitkNow.GetSize()[2]
        '''
        z_img = img.GetSize()[2]
                z_seg = seg.GetSize()[2]
                if z_img < 105:
                    print('This is an incomplete scan!')
                    bad_scans.append(seg_id)
                else:
                    if z_img > 200:
                        img = crop_full_body(img, int(z_img * 0.65))
                        seg = crop_full_body(seg, int(z_seg * 0.65))
        ''' 
        ##img = crop_full_body(imgSitkNow, int(z_img * 0.65))
        img_dir = ii
        img_id = finalNameNow
        
            
        print('interplolate')
        try: 
            img_interp = interpolate( 
                patient_id=img_id,
                path_to_nrrd=img_dir, 
                interpolation_type='linear', #"linear" for image
                new_spacing=(1, 1, 3), 
                return_type='sitk_obj',
                output_dir='',
                image_format=image_format)
            
            xx,yy,zz = img_interp.GetSize()
            if zz > 200:
                print("INSIDE")
                new_zz = 200
                img_interp = crop_upper_body(img_interp, 0, new_zz)
            
            
            reg_img, fixed_img, moving_img, final_transform = nrrd_reg_rigid( 
                patient_id=img_id, 
                moving_img=img_interp, 
                output_dir='', 
                fixed_img=fixed_img, 
                image_format=image_format)
        
            img_RegCrop_dir = datasetPath_out
            ##reg_img = img_interp
            
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
