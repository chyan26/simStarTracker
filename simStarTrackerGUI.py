# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import os
from PyQt5 import QtCore, QtGui, QtWidgets
import logging
from astropy.io import fits
from astropy import wcs
import numpy as np
import pandas as pd
import math
import time
from PIL import Image

IMAGE_WIDTH = 1096.0
IMAGE_HEIGHT = 2048.0
logging.basicConfig(level=logging.INFO)

def polar(x, y):
    theta = np.arctan2(y, x)* 180 / np.pi
    rho = np.hypot(x, y)
    
    for i in range(len(theta)):
        if theta[i]<0: 
            theta[i]=theta[i]+360
    
    #ind=np.where(theta<0)

    #theta[ind]=theta[ind]+360
    return theta, rho


class simStarTrackerImage(object):
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

    def buildWCS(self,wcsFile=None):
        if wcsFile is not None:
            logging.info(f'Buiding WCS from file')

            hdulist = fits.open(wcsFile)
            w = wcs.WCS(hdulist[0].header,naxis=2)
        else:
            w = wcs.WCS(naxis=2)

       
            w.wcs.crpix = [self.xpix,self.ypix]
            #w.wcs.cdelt = np.array([-0.018138888, -0.018138888])
            #print(f'Rot = {rotation}')
            #w.wcs.crota = [0,rotation]
            w.wcs.cd = np.array([[0.0, -0.018138888], [0.018138888,0.0]])
            w.wcs.crval = [self.ra,self.dec]
            w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
            #print(w.wcs)
            # Apply rotation to the WCS
            if self.rot is not None:
                rotation = 0
            else:
                rotation = self.rot
                w.wcs.crota = [rotation, rotation]
                logging.info(f'Rotate WCS by {rotation} degrees.')
        
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
        
        df['Nphoton']=factor*(df['Jy']*10e-26/e_photon)*frequency*psize*qe*tran*self.exptime/gain
        
        # Selecting stars in the field
        stars=df[(50 < df['x']) & (df['x'] <(self.imgwidth)-20) & (df['y'] > 50) & (df['y'] < (self.imgheight)-20)]

        self.starDF = stars
        
        logging.info(f'Star catalog is ready')
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
                if i == 0:
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
            Image.fromarray(self.tiffimage).save(f'{self.outputFileName}')
            #cv2.imwrite(f'{self.outputFileName}', self.tiffimage)
        else:
            Image.fromarray(self.tiffimage).save(f'simStarTracker_{day}.tiff')
            #cv2.imwrite(f'simStarTracker_{day}.tiff',self.tiffimage)        

        logging.info(f'TIFF is saved as {self.outputFileName}')
        #pass

    



class Ui_MainWindow(object):
    def __init__(self) -> None:
        super().__init__()
        self.ra =None
        self.dec = None
        self.rot = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("simStarTracker")
        MainWindow.resize(500, 450)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        
    
        # Boresight Label
        self.boresightLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.boresightLabel.setFont(font)
        self.boresightLabel.setObjectName("boresightLabel")
        self.gridLayout.addWidget(self.boresightLabel, 0, 0, 1, 4)
        
        spacerItem1 = QtWidgets.QSpacerItem(501, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 4, 1, 12)
        
        # Boresight setting
        self.RAlabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.RAlabel.setFont(font)
        self.RAlabel.setObjectName("RAlabel")
        self.gridLayout.addWidget(self.RAlabel, 1, 0, 1, 1)
        
        self.RAlineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.RAlineEdit.setObjectName("RAlineEdit")
        self.gridLayout.addWidget(self.RAlineEdit, 1, 1, 1, 5)

        self.DecLabel = QtWidgets.QLabel(self.centralwidget)
        self.DecLabel.setObjectName("DecLabel")
        self.gridLayout.addWidget(self.DecLabel, 1, 6, 1, 1)

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 7, 1, 4)
        
        self.RotLabel = QtWidgets.QLabel(self.centralwidget)
        self.RotLabel.setObjectName("RotLabel")
        self.gridLayout.addWidget(self.RotLabel, 1, 11, 1, 2)
        
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 13, 1, 2)
        
        # Image centers
        self.xcentLabel = QtWidgets.QLabel(self.centralwidget)
        self.xcentLabel.setObjectName("xcentLabel")
        self.gridLayout.addWidget(self.xcentLabel, 3, 0, 1, 3)

        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 3, 3, 1, 5)

        self.ycentLabel = QtWidgets.QLabel(self.centralwidget)
        self.ycentLabel.setObjectName("ycentLabel")
        self.gridLayout.addWidget(self.ycentLabel, 3, 8, 1, 2)
        
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 10, 1, 4)

        self.lineEdit_3.setText(f' {IMAGE_WIDTH/2}')
        self.lineEdit_4.setText(f' {IMAGE_HEIGHT/2}')

        # Image setting Label
        self.ImageSettingLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.ImageSettingLabel.setFont(font)
        self.ImageSettingLabel.setObjectName("ImageSettingLabel")
        self.gridLayout.addWidget(self.ImageSettingLabel, 2, 0, 1, 5)
        
        spacerItem2 = QtWidgets.QSpacerItem(476, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 2, 5, 1, 11)
        
        # Image width and height
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 2)

        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout.addWidget(self.lineEdit_5, 4, 2, 1, 6)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 8, 1, 1)

        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.lineEdit_6, 4, 9, 1, 5)

        self.lineEdit_5.setText(f' {IMAGE_WIDTH}')
        self.lineEdit_6.setText(f' {IMAGE_HEIGHT}')

        # Setting EXPTIME 
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 5, 0, 1, 3)

        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout.addWidget(self.lineEdit_7, 5, 3, 1, 5)
        self.lineEdit_7.setText(f' 0.1')
        
        spacerItem = QtWidgets.QSpacerItem(394, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 5, 8, 1, 8)


        # WCS file selection
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 6, 0, 1, 3)

        self.wcslineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.wcslineEdit.setObjectName("wcslineEdit")
        self.gridLayout.addWidget(self.wcslineEdit, 6, 3, 1, 9)

        self.selectButton = QtWidgets.QPushButton(self.centralwidget)
        self.selectButton.setObjectName("selectButton")
        self.gridLayout.addWidget(self.selectButton, 6, 12, 1, 2)
        self.selectButton.clicked.connect(self.openWCSNameDialog)

        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setObjectName("loadButton")
        self.gridLayout.addWidget(self.loadButton, 6, 14, 1, 1)
        self.loadButton.clicked.connect(self.loadWCSButtonSlot)


        # Option Label
        self.optionLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
    
        self.optionLabel.setFont(font)
        self.optionLabel.setObjectName("optionLabel")
        self.gridLayout.addWidget(self.optionLabel, 7, 0, 1, 5)


        
        # Adding sky noise        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 8, 0, 1, 4)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout.addWidget(self.lineEdit_9, 8, 4, 1, 5)
        self.lineEdit_9.setText(' 5.0')    

        spacerItem3 = QtWidgets.QSpacerItem(343, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 8, 9, 1, 7)
        
        
        # Adding Quit and Go button
        self.quitButton = QtWidgets.QPushButton(self.centralwidget)
        self.quitButton.setObjectName("quitButton")
        self.quitButton.clicked.connect(QtWidgets.QApplication.instance().quit)
        self.gridLayout.addWidget(self.quitButton, 10, 12, 1, 2)
        self.goButton = QtWidgets.QPushButton(self.centralwidget)
        self.goButton.setObjectName("goButton")
        self.gridLayout.addWidget(self.goButton, 10, 14, 1, 2)
        self.goButton.clicked.connect(self.goButtonSlot)
        
        
        

        # Radio button
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tiffButton = QtWidgets.QRadioButton(self.groupBox)
        self.tiffButton.setObjectName("select_tiff")
        self.horizontalLayout.addWidget(self.tiffButton)
        self.fitsButton = QtWidgets.QRadioButton(self.groupBox)
        self.fitsButton.setObjectName("select_fits")
        self.horizontalLayout.addWidget(self.fitsButton)
        self.gridLayout.addWidget(self.groupBox, 9, 0, 1, 8)
        self.tiffButton.setChecked(True)

        # Adding output file name line text
        
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 9, 8, 1, 3)

        self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.gridLayout.addWidget(self.lineEdit_10, 9, 11, 1, 4)
        self.browse = QtWidgets.QPushButton(self.centralwidget)

        self.browse.setObjectName("browse")
        self.gridLayout.addWidget(self.browse, 9, 15, 1, 1)
        self.browse.clicked.connect(self.openFileNameDialog)

        
        
        # Make GUI here
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 639, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.optionLabel.setText(_translate("MainWindow", "Options"))
        self.ycentLabel.setText(_translate("MainWindow", "Y Center"))
        self.RAlabel.setText(_translate("MainWindow", "R.A."))
        self.label_4.setText(_translate("MainWindow", "Width"))
        self.label_5.setText(_translate("MainWindow", "Height"))
        self.label.setText(_translate("MainWindow", "Exp Time"))
        self.boresightLabel.setText(_translate("MainWindow", "Boresight"))
        self.label_2.setText(_translate("MainWindow", "WCS File"))
        self.DecLabel.setText(_translate("MainWindow", "Dec"))
        self.ImageSettingLabel.setText(_translate("MainWindow", "Image Setting"))
        self.selectButton.setText(_translate("MainWindow", "Select"))
        self.RotLabel.setText(_translate("MainWindow", "Rotation"))
        self.xcentLabel.setText(_translate("MainWindow", "X Center"))
        self.quitButton.setText(_translate("MainWindow", "Quit"))
        self.goButton.setText(_translate("MainWindow", "Go !"))
        self.browse.setText(_translate("MainWindow", "browse"))
        self.loadButton.setText(_translate("MainWindow", "Load"))
        self.label_3.setText(_translate("MainWindow", "Sky Noise"))
        self.label_6.setText(_translate("MainWindow", "File Name"))
        self.groupBox.setTitle(_translate("MainWindow", "Output Format"))
        self.tiffButton.setText(_translate("MainWindow", "TIFF"))
        self.fitsButton.setText(_translate("MainWindow", "FITS"))

    def openWCSNameDialog(self):    
        options = QtWidgets.QFileDialog.Options()
        #options |= QtWidgets.QFileDialog.DontUseNativeDialog
        wcsName, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Select WCS file", QtCore.QDir.currentPath(),
            "All Files (*.wcs)", options=options)
        if wcsName:
            self.wcslineEdit.setText(wcsName)
            logging.info(f'WCS name ={wcsName}')

    def loadWCSButtonSlot(self):
        wcsName = self.wcslineEdit.text()
        if wcsName:
            hdulist = fits.open(wcsName)
            w = wcs.WCS(hdulist[0].header,naxis=2)
            logging.info('Using WCS from file')

            self.lineEdit_2.setText(f'{np.rad2deg(np.arcsin(-w.wcs.cd[1,0])) % 90:6.2f}')
            self.RAlineEdit.setText(f'{w.wcs.crval[0]:15.8f}')
            self.lineEdit.setText(f'{w.wcs.crval[1]:15.8f}')
            
            self.ra = w.wcs.crval[0]
            self.dec = w.wcs.crval[1]
            self.rot = np.rad2deg(np.arcsin(-w.wcs.cd[1,0])) % 90

    def openFileNameDialog(self):    
        options = QtWidgets.QFileDialog.Options()
        #options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Select Output file", QtCore.QDir.currentPath(),
            "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_10.setText(fileName)
            logging.info(f'output filename ={fileName}')

    def goButtonSlot(self):
        skynoise = self.lineEdit_9.text()
        outputFile =  self.lineEdit_10.text()
        
        st = simStarTrackerImage(IMAGE_WIDTH,IMAGE_HEIGHT)
        st.ra = self.ra
        st.dec = self.dec
        st.rot = self.rot
        st.skynoise = float(skynoise)
        etime = self.lineEdit_7.text()
        if etime:
            st.exptime = float(etime)
            logging.info(f'Exptime ={st.exptime}')
        
        # Build WCS
        if self.wcslineEdit.text():
            st.buildWCS(wcsFile=self.wcslineEdit.text())

            catalog = './Catalog/gsc.all'
            stars=st.readStarCatalog(catalog)
            st.makeTiFFimage(stars)

        if outputFile:
            st.outputFileName=outputFile
            logging.info(f'output file ={outputFile}')
            st.writeTIFFimage()

if __name__ == "__main__":
    import sys
    

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    
    
    sys.exit(app.exec_())
