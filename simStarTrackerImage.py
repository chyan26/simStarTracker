
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import logging
from astropy.io import fits
from astropy import wcs
import numpy as np
import pandas as pd
import math
import time
import cv2

IMAGE_WIDTH = 1096.0
IMAGE_HEIGHT = 2048.0

def polar(x, y):
    theta = np.arctan2(y, x)* 180 / np.pi
    rho = np.hypot(x, y)
    
    for i in range(len(theta)):
        if theta[i]<0: 
            theta[i]=theta[i]+360
    
    #ind=np.where(theta<0)

    #theta[ind]=theta[ind]+360
    return theta, rho


class SimStarTrackerImage(object):
    def __init__(self, imgwidth, imgheight):
        self.imgwidth = int(imgwidth)
        self.imgheight = int(imgheight)
        self.xpix = imgwidth/2
        self.ypix = imgheight/2
        self.wcs = None
        self.ra = None
        self.dec = None
        self.exptime = None
        self.rot=None
        self.starDF = None
        self.outputFileName = None

        self.skynoise = 0
        self.tiffimage = None

        logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s [%(filename)s:%(lineno)d] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
        self.logger = logging.getLogger('simStarTracker')
        self.logger.setLevel(logging.INFO)

    def buildWCS(self,wcsFile=None):
        if wcsFile is not None:
            self.logger.info(f'Buiding WCS from file')

            hdulist = fits.open(wcsFile)
            w = wcs.WCS(hdulist[0].header,naxis=2)
        else:
            self.logger.info(f'Buiding WCS from empty')
            w = wcs.WCS(naxis=2)

       
            w.wcs.crpix = [self.xpix,self.ypix]
            #w.wcs.cdelt = np.array([-0.018138888, -0.018138888])
            #print(f'Rot = {rotation}')
            #w.wcs.crota = [0,rotation]
            scale = 0.0183333
            #w.wcs.cd = np.array([[0.0, -0.018138888], [0.018138888,0.0]])
            w.wcs.crval = [self.ra,self.dec]
            w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
            #print(w.wcs)
            # Apply rotation to the WCS
            if self.rot is None:
                rotation = 0
            else:
                rotation = self.rot
                w.wcs.cd=scale * np.array([[np.cos(np.deg2rad(rotation)), np.sin(np.deg2rad(rotation))],
                    [-np.sin(np.deg2rad(rotation)),np.cos(np.deg2rad(rotation))]])


                w.wcs.crota = [rotation, rotation]
                self.logger.info(f'Rotate WCS by {rotation} degrees.')
        
        self.wcs = w    

    def readStarCatalog(self,catalog):
        
        df = pd.read_csv(catalog, sep='\s+')

        # Convert them into pixel cooridnate
        a=map(list,zip(*[df['ra'].values,df['dec'].values]))
        coord=np.array([])
        for _ in a:
            if coord.size == 0:
                coord = np.array([_])
            else:
                coord = np.vstack((coord,_))

        pixcrd2 = self.wcs.wcs_world2pix(coord, 1)

        # Putting X, Y to data frame
        df['x']=pixcrd2[:,0]
        df['y']=pixcrd2[:,1]

        # Converting Vmag to ABmag. Note the lambda = 0.5456 micron 
        df['ABmag']=0.02+df['Vmag']
        df['Jy']=10**((df['ABmag']-8.9)/(-2.5))
        df['theta'],df['rho']=polar(df['x'].values-self.xpix,df['y'].values-self.ypix)


        # This is the photon energy at 0.5456 micron 
        e_photon = 3.61e-19 # J 

        # Assuming the bandwidth from 0.35 to 1 um
        frequency = 401.75e12 # Hz

        # Collecting area, the aperture is 1 cm = 0.01 m.  
        psize = math.pi*0.005**2

        # give an average QE from 0.35 to 1 um
        qe=0.36

        # Gain, in the unit of e-/DN
        gain=10.2 

        # Transmission 
        tran =0.94

        # scaling factor for flux.  This is because V mag will over estimate 
        # the total flux.
        #factor = 0.46
        factor = 0.9
        
        if self.exptime is None:
            self.logger.info('Exptime is not given use default 0.1')
            self.exptime = 0.1

        df['Nphoton']=factor*(df['Jy']*10e-26/e_photon)*frequency*psize*qe*tran*self.exptime/gain
        
        # Selecting stars in the field
        
        stars=df[(50 < df['x']) & (df['x'] <(self.imgwidth)-20) & (df['y'] > 50) & (df['y'] < (self.imgheight)-20)]

        self.starDF = stars
        
        self.logger.info(f'Star catalog is ready')
        return stars

    def getRGBvalue(self,photon):
    
        if photon < 1:
            return 0,0,0
        else:
            count = [e for e in range(256)]
            r=[]
            g=[]
            b=[]


            for i in count:
                if i is 0:
                    r.append(0)
                    g.append(0)
                    b.append(0)
                elif i < 50:
                    r.append(10**(-3.358+2.878*np.log10(i)))
                    g.append(10**(-3.582+3.016*np.log10(i)))
                    b.append(10**(-5.943+4.539*np.log10(i)))
                else:
                    r.append(10**(-2.175+2.138*np.log10(i)))
                    g.append(10**(-2.206+2.167*np.log10(i)))
                    b.append(10**(-2.433+2.207*np.log10(i)))

            # scale the count, so that the maximum of R+G+B = 2048
            d = {'count': count, 
                'R':2048*(r/(r[255]+g[255]+g[255])), 
                'G':2048*(g/(r[255]+g[255]+g[255])), 
                'B':2048*(b/(r[255]+g[255]+g[255]))}
            df = pd.DataFrame(data=d)

            idx=np.random.uniform(0, 255.0,3)
            while abs(photon-df["R"][idx[0].astype(int)]-df["G"][idx[1].astype(int)]-df["B"][idx[2].astype(int)]) > 1:
                idx=np.random.uniform(0, 255,3)

            #print(photon-df["R"][idx[0].astype(int)]-df["G"][idx[1].astype(int)]-df["B"][idx[2].astype(int)])
            #print(f'{rid} {df["R"][rid]} {gid} {df["G"][gid]} {bid} {df["B"][bid]} ')

            return idx[0].astype(int),idx[1].astype(int),idx[2].astype(int)

    def makeTiFFimage(self, stars):
        
        skyimage=np.zeros((self.imgheight, self.imgwidth))

        if os.path.isfile('rgb_table.npy'):
            rgb_value=np.load('rgb_table.npy')
        else:
            rgb_value=[]
            for i in range(1280):
                rgb_value.append(self.getRGBvalue(i))
            np.save('rgb_table', rgb_value)

        stars.insert(len(stars.columns),'distx',stars['x'],True)
        stars.insert(len(stars.columns),'disty',stars['y'],True)

        for index, row in stars.iterrows():
            #center = SkyCoord(dec=dec, ra=ra, frame='icrs',unit='deg')
            #target = SkyCoord(dec=row['dec'], ra=row['ra'], frame='icrs',unit='deg')
        
        
            x=np.floor(row['distx']).astype(int)
            y=np.floor(row['disty']).astype(int)
            skyimage[y,x]=skyimage[y,x]+row['Nphoton']
        
        #except:
        img = np.zeros((skyimage.shape[0], skyimage.shape[1], 3), dtype = "uint8")
        logging.info(f'Sky noise ={self.skynoise}')

        img[:,:,0]=np.random.normal(self.skynoise, 1.0, 
            self.imgheight*self.imgwidth).reshape(self.imgheight,self.imgwidth)
        
        tiffimage = skyimage/np.max(skyimage)*1279
        #print(np.max(skyimage))
        for i in range(skyimage.shape[0]):
            for j in range(skyimage.shape[1]):
                if tiffimage[i, j] > 1279:
                    tiffimage = 1279
                else:
                    skyvalue = int(tiffimage[i, j])

                img[i,j,0]=rgb_value[skyvalue][2]
                img[i,j,1]=rgb_value[skyvalue][1]
                img[i,j,2]=rgb_value[skyvalue][0]
        
        self.tiffimage= img

    def writeTIFFimage(self):
        day = time.strftime('%Y%m%d')

        if self.outputFileName is not None:
            cv2.imwrite(f'{self.outputFileName}', self.tiffimage)
        else:
            cv2.imwrite(f'simStarTracker_{day}.tiff',self.tiffimage)        

        self.logger.info(f'TIFF is saved as {self.outputFileName}')
        #pass

    