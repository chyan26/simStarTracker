import pandas as pd
import random
import numpy as np
import math
import sys
import argparse
import math
import sep
from PIL import Image
import cv2
import time

from numpy.lib import recfunctions as rfn


from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord

def findSpot(data, sigma):
    image=data
    #m, s = np.mean(image), np.std(image)
    bkg = sep.Background(image, bw=32, bh=32, fw=3, fh=3)
    objs = sep.extract(image-bkg, sigma, err=bkg.globalrms)
    aper_radius=3
    
    # Calculate the Kron Radius
    kronrad, krflag = sep.kron_radius(image, objs['x'], objs['y'], \
        objs['a'], objs['b'], objs['theta'], aper_radius)

    r_min = 3
    use_circle = kronrad * np.sqrt(objs['a'] * objs['b'])
    cinx=np.where(use_circle <= r_min)
    einx=np.where(use_circle > r_min)

    # Calculate the equivalent of FLUX_AUTO
    flux, fluxerr, flag = sep.sum_ellipse(image, objs['x'][einx], objs['y'][einx], \
        objs['a'][einx], objs['b'][einx], objs['theta'][einx], 2.5*kronrad[einx],subpix=1)		

    cflux, cfluxerr, cflag = sep.sum_circle(image, objs['x'][cinx], objs['y'][cinx],
                                    objs['a'][cinx], subpix=1)

    # Adding half pixel to measured coordinate.  
    objs['x'] =  objs['x']+0.5
    objs['y'] =  objs['y']+0.5

    objs['flux'][einx]=flux
    objs['flux'][cinx]=cflux


    r, flag = sep.flux_radius(image, objs['x'], objs['y'], \
        6*objs['a'], 0.3,normflux=objs['flux'], subpix=5)

    flag |= krflag
 
    objs=rfn.append_fields(objs, 'r', data=r, usemask=False)

    objects=objs[:]
    
    return objects

def polar(x, y):
    theta = np.arctan2(y, x)* 180 / np.pi
    rho = np.hypot(x, y)
    
    for i in range(len(theta)):
        if theta[i]<0: 
            theta[i]=theta[i]+360
    
    #ind=np.where(theta<0)

    #theta[ind]=theta[ind]+360
    return theta, rho



def readPSFfromFile(file):
    
    with open(file,'r') as f:
        lines = f.readlines()[19:]

    image=np.array([])
    for l in lines:
        if image.size == 0:
            image = np.array(l.replace("\n","").split()).astype(float)
        else:
            image = np.vstack((image, np.array(l.replace("\n","").split()).astype(float)))
    
    image=image/np.sum(image)
    return image

def getPSFfromImage(theta, rho):
    hdul = fits.open('psf.fits')
    image=hdul[0].data
    objs=findSpot(image.astype(np.float32),30)
    psf = pd.DataFrame(data={'x':np.floor(objs['x']).astype(int),
                             'y':np.floor(objs['y']).astype(int),
                             'r':np.floor(8*objs['r']).astype(int)})
    psf['x0']=psf['x']-psf['r']
    psf['x1']=psf['x']+psf['r']
    psf['y0']=psf['y']-psf['r']
    psf['y1']=psf['y']+psf['r']
    
    
    psf['theta'],psf['rho']=polar(psf['x'].values-822,psf['y'].values-1920)
    
    # selecting closet PSF
    ind=np.where((psf['rho']-rho).abs() == min((psf['rho']-rho).abs()))
    
    # Getting sub-image
    x0=psf['x0'][ind[0]].values[0]
    x1=psf['x1'][ind[0]].values[0]
    y0=psf['y0'][ind[0]].values[0]
    y1=psf['y1'][ind[0]].values[0]
    
    subimage=image[y0:y1,x0:x1]

    
    
    # calculating the angle difference
    delta_angle = psf['theta'][ind[0]] - theta
    
    #print(psf['theta'][ind[0]],psf['rho'][ind[0]], delta_angle.values[0])


    #Rotate the subimage
    rot_subimage=np.array(
        Image.fromarray(subimage).rotate(-delta_angle.values[0], resample=Image.BILINEAR)
        )
    
    # Cast the array to float and normalize to unity
    
    psfimg=rot_subimage.astype(np.float32)
    psfimg[:]=psfimg[:]-np.median(psfimg)
    psfimg[psfimg < 0]=0
    psfimg[:]=psfimg[:]/np.sum(psfimg)

    #print(psfimg)
    return psfimg

def getRGBvalue(photon):
    
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



def saveTIFFImage(image, filename):


    pass

def readStarCatalog(catalog):
    df = pd.read_csv(catalog, sep='\s+')

    return df

def main():
    


    parser = argparse.ArgumentParser(description='Generating simulation image for Star Tracker.')

    #parser.add_argument('-l','--loop',default=False, dest="loop",
    #                    help='Number of simulated image.')
    parser.add_argument('-p','--nopsf',default=False, dest="nopsf",
                        help='Removing PSF from star spots',action="store_true")
    parser.add_argument('-s','--savetif',default=False, dest="savetif",
                        help='Save TIFF file for cell phone display',action="store_true")
    parser.add_argument('-v','--verbose',default=False, dest="verbose",
                        help='Display detailed information',action="store_true")
    parser.add_argument('-r','--ra',default=False,type=float, dest="ravalue",
                        help='Setting RA value.')
    parser.add_argument('-d','--dec',default=False, dest="decvalue",
                        help='Setting DEC value.',type=float)
    parser.add_argument('-e','--etime',default=False, dest="etime",
                        help='Setting exposure time.',type=float)
    parser.add_argument('-c','--csv',default=False, dest="csvname",
                        help='Exporting CSV file.',type=str)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help() 
        sys.exit(1) 
    

    catalog = './Catalog/gsc.all'
    df=readStarCatalog(catalog)

    if args.ravalue is False:
        ra = random.uniform(0, 360)
        print(f'Using random for RA')
    else:
        ra = ravalue
    
    if args.decvalue is False:
        print(f'Using random for Dec')
        dec = random.uniform(-90, 90)
    else:
        dec = decvalue
    
    
    ra = 1.64066278187
    dec = 28.713430003
    #ra = random.uniform(-90, 90)
    #dec = random.uniform(0, 360)
    
    if args.verbose:
        print(f'Boresight center is RA = {ra} DEC ={dec}')

    # Give a exposure time
    if args.etime is False:
        exptime = 0.1 #
    else:
        exptime = etime

    if args.verbose:
        print(f'Exposure time ={exptime}')

    a=map(list,zip(*[df['dec'].values,df['ra'].values]))
    coord=np.array([])
    for _ in a:
        if coord.size == 0:
            coord = np.array([_])
        else:
            coord = np.vstack((coord,_))

    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)

    # Set up an orthographic projection
    # Vector properties may be set with Python lists, or Numpy arrays
    w.wcs.lonpole = 180
    w.wcs.latpole = 0
    w.wcs.crpix = [640.5,512.5]
    w.wcs.cdelt = np.array([-0.0178552972045, 0.0178617816256])
    w.wcs.cd = np.array([[0.000436314221715, -0.0178552972045], [0.0178617816256,0.000389544945898]])
    w.wcs.crval = [ra,dec]
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]

    # Convert the same coordinates to pixel coordinates.
    pixcrd2 = w.wcs_world2pix(coord, 1)
    #print(pixcrd2)

    # Putting X, Y to data frame
    df['x']=pixcrd2[:,0]
    df['y']=pixcrd2[:,1]

    # Converting Vmag to ABmag. Note the lambda = 0.5456 micron 
    df['ABmag']=0.02+df['Vmag']
    df['Jy']=10**((df['ABmag']-8.9)/(-2.5))
    df['theta'],df['rho']=polar(df['x'].values-640,df['y'].values-512)


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
    factor = 0.5

    df['Nphoton']=factor*(df['Jy']*10e-26/e_photon)*frequency*psize*qe*tran*exptime/gain

    # Now, write out the WCS object as a FITS header
    header = w.to_header()

    # header is an astropy.io.fits.Header object.  We can use it to create a new
    # PrimaryHDU and write it to a file.
    #hdu = fits.PrimaryHDU(header=header)

    # Selecting stars in the field
    stars=df[(50 < df['x']) & (df['x'] < 1230) & (df['y'] > 50) & (df['y'] < 974)]

    # Making a sky frame with noise 
    skyimage=np.random.normal(0.0, 1.0, 1024*1280).reshape(1024,1280)
    #skyimage=np.zeros((1024,1280))
    
    
    # Establish a table for mapping flux to RGB value
    if args.savetif is True:   
        rgb_value=[]
        for i in range(1280):
            rgb_value.append(getRGBvalue(i))

    if args.csvname is not None:
        stars.to_csv(f'{args.csvname}')

    # Loop through star table 
    for index, row in stars.iterrows():
        center = SkyCoord(dec, ra, frame='icrs',unit='deg')
        target = SkyCoord(row['dec'], row['ra'], frame='icrs',unit='deg')
        
        psf=getPSFfromImage(row['theta'],row['rho'])
        
        # sep = center.separation(target)
        # #print(sep.degree)
        # if sep.degree < 3:
        #     psf = psf0
        # if 3 <= sep.degree < 8:
        #     psf = psf5
        # if sep.degree >= 8:
        #     psf = psf8
        # #try:
        # #print(y-50,y+50,x-50,x+50)
        # #if 90 > row['Nphoton'] > 100:
        #print(row['Nphoton'],row['Vmag'])
        
         


        x=np.floor(row['x']).astype(int)
        y=np.floor(row['y']).astype(int)
        if args.nopsf is True:
            skyimage[y,x]=skyimage[y,x]+row['Nphoton']
        else:
            point = row['Nphoton']*psf
            dx=int(point.shape[0]/2)
            dy=int(point.shape[1]/2)
            skyimage[y-dy:y+dy,x-dx:x+dy]=skyimage[y-dy:y+dy,x-dx:x+dy]+point
                #skyimage[y-point.shape[0]/2:y+point.shape[0]/2,x-point.shape[1]/2:x+point.shape[1]/2]+point
        #except:
    day = time.strftime('%Y%m%d')
    
    
    skyimage[skyimage<0]=0
    skyimage[skyimage > 1024] =1024
    
    if args.savetif is True:
        img = np.zeros((skyimage.shape[0], skyimage.shape[1], 3), dtype = "uint8")

        for i in range(skyimage.shape[0]):
            for j in range(skyimage.shape[1]):
                img[i,j,0]=rgb_value[skyimage[i, j].astype('int')][2]
                img[i,j,1]=rgb_value[skyimage[i, j].astype('int')][1]
                img[i,j,2]=rgb_value[skyimage[i, j].astype('int')][0]

        cv2.imwrite(f'simStarTracker_{day}.tiff',img)        


    # Adding Poisson Noise to each pixel.
    for i in range(skyimage.shape[0]):
        for j in range(skyimage.shape[1]):
            pvalue=np.sqrt(abs(skyimage[i, j]))
            poisson_n=random.uniform(-pvalue, pvalue)
            skyimage[i, j] =  skyimage[i, j]+poisson_n

    skyimage[skyimage > 1024] =1024
    
 

        
    hdu = fits.PrimaryHDU(skyimage, header=header)
    hdu.writeto(f'simStarTracker_{day}.fits', overwrite=True)


    
        #np.set_printoptions(linewidth=1000)
 
if __name__ == '__main__':
    main()

