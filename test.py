from edgy import *
from pylab import *


lena = imread('/home/ftruzzi/ciencia/DADOS/yorkurban/frames/P1020822.jpg')
lena = flipud(lena)
a = Canny(lena)
show()


figure(1)
title("test")
imshow(a.test, cm.gist_gray)
figure(2)
title("mag")
imshow(a.mag, cm.gist_gray)
figure(3)
title("diff")
imshow(a.test-a.mag, cm.gist_gray)
