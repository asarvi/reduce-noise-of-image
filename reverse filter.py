import numpy as np
import cv2
from matplotlib import pyplot as plt
#this is reversed filter for first picture

imageG = cv2.imread('degeraded1.jpg',0)
imageH = cv2.imread('h.jpg',0)

img_float32 = np.fft.fft2(imageG)
fshift = np.fft.fftshift(img_float32)
#show the furier  image transform
furier_tr = 20*np.log(np.abs(fshift))

#picture h furie transform
img_float322 = np.fft.fft2(imageH)
fshift2 = np.fft.fftshift(img_float322)
#show the furier  image transform
furier_tr2 = 20*np.log(np.abs(fshift2))

plt.subplot(121),plt.imshow(imageG, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(furier_tr, cmap = 'gray')
plt.title('furier transform'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(imageH, cmap = 'gray')
plt.title('Input h'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(furier_tr2, cmap = 'gray')
plt.title('furier transform'), plt.xticks([]), plt.yticks([])
plt.show()

# height, width of H image
height = imageH.shape[0]
width = imageH.shape[1]



for x in range(512):
   for y in range(512):
       if(fshift2[x,y]!= 0 ):
        fshift[x,y] = (fshift[x,y]) /(fshift2[x,y])


#rebuild image after filter

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

for x in range(width):
   for y in range(height):
     print(fshift[x,y])



cv2.imwrite('output.png',img_back)
furier_tr3 = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(furier_tr3, cmap = 'gray')
plt.title('modified furier'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('output'), plt.xticks([]), plt.yticks([])

plt.show()







