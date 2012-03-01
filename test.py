from edgy import *
from pylab import *
from sys import argv


lena = imread(argv[1])
lena = flipud(lena)
a = Canny(lena)

subplot(121)
imshow(a.test,  cm.gist_gray, interpolation='nearest')
lena = imread("/home/ftruzzi/Desktop/nonmaximum1.png")
subplot(122)
imshow(lena,  cm.gist_gray, interpolation='nearest')

