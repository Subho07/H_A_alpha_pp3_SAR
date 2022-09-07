# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:44:37 2022

@author: dey_sb
"""

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

#%%
def read_bin(file):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
   
    return arr

#%%

def write_bin(file,wdata,refData):
                
    ds = gdal.Open(refData)
    [cols, rows] = wdata.shape
            
    driver = gdal.GetDriverByName("ENVI")
    outdata = driver.Create(file, rows, cols, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
                
    outdata.SetDescription(file)
    outdata.GetRasterBand(1).WriteArray(wdata)
    # outdata.GetRasterBand(1).SetNoDataValue(np.NaN)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!! 

def conv2d(a, f):
    filt = np.zeros(a.shape)
    wspad = int(f.shape[0]/2)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    filt_data = np.einsum('ij,ijkl->kl', f, subM)
    filt[wspad:wspad+filt_data.shape[0],wspad:wspad+filt_data.shape[1]] = filt_data
    return filt
#%% 
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()
   
T11 = read_bin(folder_selected + '/T11.tif')
T12_im = read_bin(folder_selected + '/T12_imag.tif')
T12_re = read_bin(folder_selected + '/T12_real.tif')
T22 = read_bin(folder_selected + '/T22.tif')

ws = 7 # window size
kernel = np.ones((ws,ws),np.float32)/(ws*ws)

T11 = conv2d(np.real(T11),kernel)
T12_im = conv2d(np.real(T12_im),kernel)
T12_re = conv2d(np.real(T12_re),kernel)
T22 = conv2d(np.real(T22),kernel)

[row,col] = np.shape(C11)

alpha_image = np.zeros([row,col])
theta_image = np.zeros([row,col])
aniso_image = np.zeros([row,col])
ent_image = np.zeros([row,col])
corr_image = np.zeros([row,col])

for ii in range(row):
    for jj in range(col):
        t11_s = T11[ii,jj]
        t12_im_s = T12_im[ii,jj]
        t12_re_s = T12_re[ii,jj]
        t22_s = T22[ii,jj]
        t12_s = t12_re_s + 1j*t12_im_s
        t21_s = np.conj(t12_s)
        
        T2= np.array([[t11_s,t12_s],
                      [t21_s,t22_s]])
        
        eigenValues, eigenVectors = np.linalg.eig(T2)
    
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
    
        p1 = eigenValues[0]/(np.sum(eigenValues))
        p2 = eigenValues[1]/(np.sum(eigenValues))
    
        alpha = np.arccos(eigenVectors[0,0]/np.sqrt(eigenVectors[0,0]**2+eigenVectors[1,0]**2))*180/np.pi
    
        alpha_ = p1*alpha+p2*(90-alpha)
        alpha_image[ii,jj] = np.real(alpha_)
        
        m_xP = np.sqrt(1 - ((4*np.linalg.det(T2))/(np.trace(T2)**2)))
        aniso_image[ii,jj] = m_xP
        
        span = np.trace(T2)
        val = (m_xP*span*(T2[0,0] - T2[1,1]))/(T2[0,0]*T2[1,1] + m_xP**2*span**2)
        theta_xP = np.rad2deg(np.arctan(np.real(val)))
        theta_image[ii,jj] = np.real(theta_xP)

        H = -(p1*np.log2(p1)+p2*np.log2(p2))
        ent_image[ii,jj] = H
    print('Row: ',ii)

#%%
infile = folder_selected + '/T11.tif'
ofilegrvi = folder_selected + '/theta_pp3.bin'
write_bin(ofilegrvi,theta_image,infile)   
ofilegrvi = folder_selected + '/aniso_pp3.bin'
write_bin(ofilegrvi,aniso_image,infile) 
ofilegrvi = folder_selected + '/ent_pp3.bin'
write_bin(ofilegrvi,ent_image,infile)
ofilegrvi = folder_selected + '/alpha_pp3.bin'
write_bin(ofilegrvi,alpha_image,infile)
