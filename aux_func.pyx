from __future__ import division
import numpy as np
import scipy.ndimage as ndimage
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
DTYPE2 = np.float64
ctypedef np.float64_t DTYPE2_t
from libcpp cimport bool
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def NMS_HySt(np.ndarray[DTYPE_t, ndim=2, mode="c"] gx, np.ndarray[DTYPE_t, ndim=2, mode="c"] gy, DTYPE2_t upperThreshold, DTYPE2_t lowerThreshold): 
  cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] mag = np.hypot(gx,gy)
  cdef np.ndarray[DTYPE2_t, ndim=2, mode="c"] out = np.zeros((gx.shape[0],gx.shape[1]))
  cdef np.ndarray[DTYPE2_t, ndim=2, mode="c"] high = np.zeros((gx.shape[0],gx.shape[1]))
  cdef np.ndarray[DTYPE2_t, ndim=2, mode="c"] low = np.zeros((gx.shape[0],gx.shape[1]))
  cdef unsigned int j, k
  cdef DTYPE2_t n
  for j in range(1,gx.shape[0]-1):
    for k in range(1,gx.shape[1]-1):     
      n = mag[j,k]
      if ( fabs(gx[j,k]) > fabs(2*gy[j,k])): 
        if ((n >= mag[j,k-1]) & (n > mag[j,k+1])): # 0
          out[j,k] = 1.0
      elif ( fabs(gy[j,k]) > fabs(2*gx[j,k])): # 90
        if ((n >= mag[j-1,k]) & (n > mag[j+1,k])):
          out[j,k] = 1.0   
      elif ( ((gy[j,k] < 2*gx[j,k]) & (gx[j,k] < 2*gy[j,k])) | (gy[j,k] > 2*gx[j,k]) & (gx[j,k] > 2*gy[j,k]) ): #135
        if ((n >= mag[j+1,k+1]) & (n > mag[j-1,k-1])):
          out[j,k] = 1.0        
      elif ( ((gx[j,k] < 2*gy[j,k]) & ( gx[j,k] < 0 )) | ((gx[j,k] > 2*gy[j,k]) & (gx[j,k] > 0)) ): #45
        if ((n >= mag[j+1,k-1]) & (n > mag[j-1,k+1])):
          out[j,k] = 1.0            
  out = out*mag
  high[np.where(out > upperThreshold)] = 1.0
  low[np.where(out > lowerThreshold)] = 1.0
  return ndimage.binary_dilation(high, iterations=-1, mask=low)

@cython.boundscheck(False)
@cython.wraparound(False)
def getInterpolatedPoint(x0,y0,x1,y1,x2,y2):
  cdef DTYPE2_t a, b, c
  a = (((y1-y0)*(x2-x0))+((y2-y0)*(x0-x1)))/(((x1**2 - x0**2)*(x2-x0))+((x2**2-x0**2)*(x0-x1)))
  b = ((y1-y0)/(x1-x0))-(x1+x0)*a
  c = y2-a*x2**2-x2*b
  return (-b/(2*a),


@cython.boundscheck(False)
@cython.wraparound(False)
def HyTresh(np.ndarray[DTYPE2_t, ndim=2, mode="c"] image not None, DTYPE2_t high_threshold, DTYPE2_t low_threshold):
  cdef unsigned int i, j
  cdef DTYPE2_t strong  = 255.0
  cdef DTYPE2_t weak = 0.0
 

  for i in xrange(1,image.shape[0]-1):
    for j in xrange(1,image.shape[1]-1):
      if (image[i,j] > high_threshold):
        image[i,j] = strong        
      elif (image[i,j] < low_threshold):
        image[i,j] = weak

  for i in xrange(1,image.shape[0]-1):
    for j in xrange(1,image.shape[1]-1):
      if ((image[i,j] < strong) and (image[i,j] != weak)):
        if ((image[i+1,j] == strong) or (image[i-1,j] == strong) or
            (image[i,j+1] == strong) or (image[i,j-1] == strong) or
            (image[i+1,j+1] == strong) or (image[i-1,j-1] == strong) or
            (image[i-1,j+1] == strong) or (image[i+1,j-1] == strong)):
          image[i,j] = strong
  return image

cdef extern double fabs(double)
cdef extern double sqrt(double)