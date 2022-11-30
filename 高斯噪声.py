import skimage.io as io
from matplotlib import pyplot as plt
import skimage


origin = io.imread('./data/0001.jpg', 0)


plt.subplot(2,2,1)
io.imshow(origin)
plt.imshow(origin, plt.cm.gray)
plt.title("origin")


plt.subplot(2,2,2)
noisy = skimage.util.random_noise(origin, mode='gaussian', var=0.5)
io.imshow(noisy)
plt.imshow(noisy, plt.cm.gray)
io.imsave('./noisy/0002.jpg', noisy)
plt.title("Gauss")


plt.subplot(2,2,3)
noisy1 = skimage.util.random_noise(origin, mode='salt')
io.imshow(noisy1)
plt.imshow(noisy1, plt.cm.gray)
io.imsave('./noisy/0003.jpg', noisy1)
plt.title("salt")


plt.subplot(2,2,4)
noisy2 = skimage.util.random_noise(origin, mode='pepper')
io.imshow(noisy2)
plt.imshow(noisy2, plt.cm.gray)
io.imsave('./noisy/0004.jpg', noisy2)
plt.title("pepper")


plt.show()