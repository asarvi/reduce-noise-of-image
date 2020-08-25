import numpy as np
import cv2
from matplotlib import pyplot as plt

#wiener filter for first image
imageG = cv2.imread('degeraded1.jpg',0)
imageH = cv2.imread('h.jpg',0)


img_float32 = np.fft.fft2(imageG)
fshift = np.fft.fftshift(img_float32)

#show the furier  image transform
furier_tr = 20*np.log(np.abs(fshift))

#picture h furie transform
img_float322 = np.fft.fft2(imageH)
fshift2 = np.fft.fftshift(img_float322
                        )
#show the furier  image transform
fshift2copy=np.fft.fftshift(img_float322 )
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
height = fshift2.shape[0]
width = fshift2.shape[1]

for x in range(width):
   for y in range(height):
       fshift2copy[x,y] = np.conj(fshift2[x,y])


for x in range(width):
   for y in range(height):
      Rw = fshift2copy[x,y] / ((fshift2copy[x,y]*fshift2[x,y])+1)
    #  Rw =np.true_divide(imageHStar[x,y],imageHStar[x,y]*imageH[x,y])
      fshift[x,y] = ((fshift[x,y]*Rw))
      print(Rw)



f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

#for x in range(width):
 #  for y in range(height):
  #   print(fshift[x,y])


cv2.imwrite('output.png',img_back)
furier_tr3 = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(furier_tr3, cmap = 'gray')
plt.title('modified furier'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('output'), plt.xticks([]), plt.yticks([])


plt.show()







