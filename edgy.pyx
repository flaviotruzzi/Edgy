import sys
import scipy.ndimage as ndimage
import scipy
import numpy as np
from pylab import imshow, cm, show, figure, hstack, vstack, zeros


class Canny:

  def __init__(self, image, sigma=1, thetaTreshold=10, upperThreshold=.3, lowerThreshold=.1, shigeru=True):
    self.upperThreshold = upperThreshold
    self.lowerThreshold = lowerThreshold
    
    self.original = image
    self.image = image
    
    if (image.shape[2] == 3):
      self.image = self.convert2Gray(self.image)

    self.smoothed = ndimage.gaussian_filter(self.image, sigma)

    self.shigeru_filter = scipy.array([
    [ -0.003776, -0.010199, 0., 0.010199, 0.003776 ],
    [ -0.026786, -0.070844, 0., 0.070844, 0.026786 ],
    [ -0.046548, -0.122572, 0., 0.122572, 0.046548 ],
    [ -0.026786, -0.070844, 0., 0.070844, 0.026786 ],
    [ -0.003776, -0.010199, 0., 0.010199, 0.003776 ]
    ])

    if shigeru:
      self.gradx = ndimage.convolve(self.image, self.shigeru_filter)
      self.grady = ndimage.convolve(self.image, self.shigeru_filter.T)
    else:
      self.gradx = ndimage.sobel(self.smoothed, 0)
      self.grady = ndimage.sobel(self.smoothed, 1)

    self.mag = np.hypot(self.gradx, self.grady)  
    self.theta = (np.arctan2(self.grady, self.gradx)+np.pi)*180.0/np.pi

    self.horizontal = np.zeros(self.theta.shape, dtype=bool)
    self.vertical = np.zeros(self.theta.shape, dtype=bool)
    self.diagonal = np.zeros(self.theta.shape, dtype=bool)
    self.anti_diagonal = zeros(self.theta.shape, dtype=bool)

    self.horizontal[np.where(
      ((self.theta >= 0) & (self.theta < 22.5)) |
      ((self.theta >= 157.5) & (self.theta < 202.5)) |
      (self.theta >= 337.5)) ] = True
    self.vertical[np.where(
      ((self.theta >= 67.5) & (self.theta < 112.5)) |
      ((self.theta >= 247.5) & (self.theta < 292.5)))] = True
    self.diagonal[np.where(
      ((self.theta >= 22.5) & (self.theta < 67.5)) |
      ((self.theta >= 202.5) & (self.theta < 247.5)))] = True
    self.anti_diagonal[np.where(
      ((self.theta >= 112.5) & (self.theta < 157.5)) |
      ((self.theta >= 292.5) & (self.theta < 337.5)))] = True
    
    # Non maximum supression
    # Horizontal
    self.test = self.mag.copy()
    x,y = np.where(self.horizontal[1:,-1:] == True)
    self.test[np.where( (
      (self.test[x,y-1] < self.test[x,y]) | 
      (self.test[x,y+1] < self.test[x,y]) == True))] = 0

    x,y = np.where(self.vertical[1:,-1:] == True)
    self.test[np.where( (
      (self.test[x-1,y] < self.test[x,y]) | 
      (self.test[x+1,y] < self.test[x,y]) == True))] = 0



  def convert2Gray(self, image):
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]    
    return 0.21*r + 0.71*g + 0.07*b