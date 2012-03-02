from edgy import *
from pylab import *
from sys import argv


lena = imread(argv[1])
lena = flipud(lena)
a = Canny(lena,sigma=4,shigeru=True)
imshow(aux_func.NMS_HySt(a.gradx,a.grady,0,0), cm.gist_gray, interpolation='nearest')
ion()
show()