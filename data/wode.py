import numpy as np
import ImShow as I
import matplotlib.pyplot as plt
from collections import Counter
data = np.loadtxt("data.txt",delimiter=",")
y = np.loadtxt("y.txt",delimiter=",")
print (data.shape)
print (y.shape)
print (Counter(y.reshape(y.size)))
Xpic = I.tile_raster_images(X = data, img_shape=(28,28), tile_shape=(10,10))
plt.imshow(Xpic,cmap='gray')
plt.show()
one_pic = data[0].reshape(28,28)
plt.imshow(one_pic,cmap='gray')
plt.show()



