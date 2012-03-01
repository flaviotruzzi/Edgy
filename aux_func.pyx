from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
DTYPE2 = np.float64
ctypedef np.float64_t DTYPE2_t
from cpython cimport bool
cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
def NMS2(np.ndarray[DTYPE_t, ndim=2, mode="c"] magnitude not None, np.ndarray[DTYPE_t, ndim=2, mode="c"] theta not None, DTYPE_t upperThreshold):
  cdef unsigned int i, j
  output = np.zeros((magnitude.shape[0],magnitude.shape[1]))
  for i in xrange(1,magnitude.shape[0]-1):
    for j in xrange(1,magnitude.shape[1]-1):
      if (theta[i,j] == 0):
        if ( (magnitude[i,j] < magnitude[i,j+1]) | (magnitude[i,j] < magnitude[i,j-1])):
          output[i,j] = 0
        else:
          output[i,j] = magnitude[i,j]
  return output


@cython.boundscheck(False)
@cython.wraparound(False)
def NMS(np.ndarray[DTYPE2_t, ndim=2, mode="c"] image not None, np.ndarray[DTYPE2_t, ndim=2, mode="c"] output not None , np.ndarray[DTYPE2_t, ndim=2, mode="c"] theta not None, DTYPE2_t upperThreshold):
  cdef unsigned int i, j
  for i in xrange(1,image.shape[0]-1):
    for j in xrange(1, image.shape[1]-1):
      if (image[i,j] > upperThreshold):
        if (theta[i,j] == 90):
          if ((image[i,j] > image[i+1,j]) and (image[i,j] > image[i-1,j])):
            output[i,j] = 255
        elif (theta[i,j] == 0):
          if ((image[i,j] > image[(i,j+1)]) and (image[i,j] > image[(i,j-1)])):
            output[i,j] = 255
        elif (theta[i,j] == 135):
          if ((image[i,j] > image[(i-1,j-1)]) and (image[i,j] > image[(i+1,j+1)])):
            output[i,j] = 255
        elif (theta[i,j] == 45):
          if ((image[i,j] > image[(i-1,j+1)]) and (image[i,j] > image[(i+1,j-1)])):
            output[i,j] = 255
  return image

@cython.boundscheck(False)
@cython.wraparound(False)
def NMS(np.ndarray[DTYPE_t, ndim=2, mode="c"] image not None, np.ndarray[DTYPE2_t, ndim=2, mode="c"] output not None , np.ndarray[DTYPE_t, ndim=2, mode="c"] theta not None, DTYPE2_t upperThreshold):
  cdef unsigned int i, j
  for i in xrange(1,image.shape[0]-1):
    for j in xrange(1, image.shape[1]-1):
      if (image[i,j] > upperThreshold):
        if (theta[i,j] == 90):
          if ((image[i,j] > image[i+1,j]) or (image[i,j] > image[i-1,j])):
            output[i,j] = 255
        elif (theta[i,j] == 0):
          if ((image[i,j] > image[(i,j+1)]) or (image[i,j] > image[(i,j-1)])):
            output[i,j] = 255
        elif (theta[i,j] == 135):
          if ((image[i,j] > image[(i-1,j-1)]) or (image[i,j] > image[(i+1,j+1)])):
            output[i,j] = 255
        elif (theta[i,j] == 45):
          if ((image[i,j] > image[(i-1,j+1)]) or (image[i,j] > image[(i+1,j-1)])):
            output[i,j] = 255
  return image

@cython.boundscheck(False)
@cython.wraparound(False)
def HyTresh(np.ndarray[DTYPE2_t, ndim=2, mode="c"] image not None, DTYPE2_t high_threshold, DTYPE2_t low_threshold):
  cdef unsigned int i, j
  cdef DTYPE2_t strong  = 255.0
  cdef DTYPE2_t weak = 0.0
  cdef bool flag = True
#  cdef np.ndarray[DTYPE2_t, ndim=output] 2 = np.zeros(image.shape[0],image.shape[1])

  for i in xrange(1,image.shape[0]-1):
    for j in xrange(1,image.shape[1]-1):
      if (image[i,j] > high_threshold):
        image[i,j] = strong        
      elif (image[i,j] < low_threshold):
        image[i,j] = weak

 # while (flag):
  #  flag = False
  for i in xrange(1,image.shape[0]-1):
    for j in xrange(1,image.shape[1]-1):
      if ((image[i,j] < strong) and (image[i,j] != weak)):
        if ((image[i+1,j] == strong) or (image[i-1,j] == strong) or
            (image[i,j+1] == strong) or (image[i,j-1] == strong) or
            (image[i+1,j+1] == strong) or (image[i-1,j-1] == strong) or
            (image[i-1,j+1] == strong) or (image[i+1,j-1] == strong)):
          image[i,j] = strong
  return image