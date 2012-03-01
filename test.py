from edgy import *
from pylab import *
from sys import argv


lena = imread(argv[1])
lena = flipud(lena)
a = Canny(lena)

horizontal = zeros(a.image.shape)
vertical = zeros(a.image.shape)
diag = zeros(a.image.shape)
antdiag = zeros(a.image.shape)

horizontal[where(a.theta2 == 0)] = 255*a.test[where(a.theta2 == 0)]
vertical[where(a.theta2 == 90)] = 255*a.test[where(a.theta2 == 90)]
diag[where(a.theta2 == 45)] = 255*a.test[where(a.theta2 == 45)]
antdiag[where(a.theta2 == 135)] = 255*a.test[where(a.theta2 == 135)]


figure(1)
subplot(221)
title("horizontal")
imshow(horizontal, cm.gist_gray, interpolation='nearest')
subplot(222)
title("vertical")
imshow(vertical, cm.gist_gray, interpolation='nearest')
subplot(223)
title("diag")
imshow(diag, cm.gist_gray, interpolation='nearest')
subplot(224)
title("antdiag")
imshow(antdiag, cm.gist_gray, interpolation='nearest')

figure(2)
subplot(121)
imshow(a.test, cm.gist_gray, interpolation='nearest')
subplot(122)
imshow(a.test2, cm.gist_gray, interpolation='nearest')

ion()
show()