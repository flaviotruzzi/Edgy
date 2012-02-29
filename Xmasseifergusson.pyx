import sys
import numpy as np
from scipy import ndimage
import scipy
from pylab import imshow, cm, show, figure, hstack, vstack, zeros



class canny:

  def __init__(self, filename, upperThreshold=50, lowerThreshold=20,
              filter='shigeru'):
    self.upperThreshold = upperThreshold
    self.lowerThreshold = lowerThreshold

    self.image = ndimage.imread(filename, flatten=True)
    self.blurred = ndimage.gaussian_filter(self.image, 1.4)

    self.shigeru_filter = scipy.array([
    [ -0.003776, -0.010199, 0., 0.010199, 0.003776 ],
    [ -0.026786, -0.070844, 0., 0.070844, 0.026786 ],
    [ -0.046548, -0.122572, 0., 0.122572, 0.046548 ],
    [ -0.026786, -0.070844, 0., 0.070844, 0.026786 ],
    [ -0.003776, -0.010199, 0., 0.010199, 0.003776 ]
    ])

    self.gradX = scipy.zeros(self.image.shape, dtype=np.float32)
    self.gradY = scipy.zeros(self.image.shape, dtype=np.float32)

    if filter == 'shigeru':
      self.gradX = ndimage.convolve(self.blurred, self.shigeru_filter.T)
      self.gradY = ndimage.convolve(self.blurred, self.shigeru_filter)      
    else:
      self.gradX = ndimage.sobel(self.blurred, 0)
      self.gradY = ndimage.sobel(self.blurred, 1)

    self.mag = np.hypot(self.gradX, self.gradY)
    self.theta = np.arctan2(self.gradY,self.gradX)

    self.v = (self.mag[1:-1] > self.mag[2:,:]) & (self.mag[1:-1,:] > self.mag[:-2,:])
    self.h = (self.mag[:,1:-1] > self.mag[:,2:]) & (self.mag[:,1:-1] > self.mag[:,:-2])
    self.d = abs(self.gradX) > abs(self.gradY)
    self.h = hstack((zeros((480,1),dtype=bool),self.h,zeros((480,1),dtype=bool)))
    self.v = vstack((zeros((1,640),dtype=bool),self.v,zeros((1,640),dtype=bool)))

    self.mask = (np.invert(self.d)*self.h + self.d*self.v)

    self.result = zeros(self.mask.shape)
    self.result[np.where(self.mask == True)] = self.mag[np.where(self.mask == True)]

    self.result *= 255.0/self.result.max()

    self.result[np.where(self.result > self.upperThreshold)] = 255
    self.result[np.where(self.result < self.lowerThreshold)] = 128

    self.cannyresult = zeros(self.result.shape, dtype=bool)
    
    self.cannyresult[np.where(self.result == 255)] = True

    self.result[np.where(self.result > self.upperThreshold)] = 255
    self.result[np.where(self.result < self.lowerThreshold)] = 128 
    self.cannyresult = zeros(self.result.shape, dtype=bool) #Mark strong edges 
    self.cannyresult[np.where(self.result == 255)] = True 
    pointsToCheck = np.where(self.result == 128) 
    count = 0 
    flag = True 
    while (flag and len(pointsToCheck[0]) != 0):  
      flag = False  
      while (count <= len(pointsToCheck[0])):  
        print pointsToCheck
        xp = pointsToCheck[count][0]  
        yp = pointsToCheck[count][1]  
        count += 1  
        if( xp-1>= 0 and xp+1 < self.result.shape[0] and yp >= 0 and yp < self.result.shape[1]):
          if (self.cannyresult[(xp-1,yp)] == True or self.cannyresult[(xp+1,yp)] or self.cannyresult[(xp,yp+1)] == True or self.cannyresult[(xp,yp-1)] == True):
            self.cannyresult[(xp,yp)] == 255
            flag = True
            pointsToCheck = np.where(self.result == 128)
            count = 0
