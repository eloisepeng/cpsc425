# Part 1 - testing
import a1
# 1
a1.boxfilter(3)

a1.boxfilter(4)

a1.boxfilter(5)

# 2
a1.gauss1d(0.3)

a1.gauss1d(0.5)

a1.gauss1d(1)

a1.gauss1d(2)

# 3
a1.gauss2d(0.5)

a1.gauss2d(1)

# 4b)
import numpy as np
from PIL import Image

im = Image.open('dog.jpg')

# convert the image to a greyscale image
im_grey = im.convert('L')
# convert to Numpy array for (for subsequent processing)
im_grey_array = np.asarray(im_grey)
a1.gaussconvolve2d(im_grey_array, 3)

# convert the image to a numpy array (for subsequent processing)
# convert the result back to a PIL image and save
im_grey = Image.fromarray(im_grey_array)

# 4c)
im.show()
im_grey.show()



