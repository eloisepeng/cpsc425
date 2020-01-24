# import packages for this assignment
from PIL import Image
import numpy as np
import math
from scipy import signal

# define global variables here
filter_sum = 1.0
lower_range, upper_range = 0, 255
scaler = 128

# Help function to check if the parameter n is odd
is_odd = lambda n: n%2 == 1

# Help function to normalize the filter
to_normalized = lambda filter: np.divide(filter, sum(filter))

# Part 1
# Q1
# Define function boxfilter(n) returns a box filter of size n by n
# Note: n is odd
def boxfilter( n ):
	"This returns a boxfilter with size n by n where n is odd"
	assert ( is_odd(n) ), "Dimension must be odd"

	# calculate value for each box
	value = filter_sum / (n * n)

	# insert values to form a numpy array of n by n
	return np.asfarray([[value] * n] * n)

# Q2	
# Define function gauss1d(sigma) returns a 1D Gaussian filter for a given value of sigma
def gauss1d(sigma):
	"This returns a 1D Gaussian filter for a given value of sigma"

	# Convert the length of Gaussian 1D array to each side of 0 as k
	# Length: 6 times sigma rounded up to the next odd integer
	# Generate a 1-D array arr of values x where the x is the distance from the center
	k = (math.ceil(6 * sigma)) // 2
	print (k)
	arr = np.arange(-k, k + 1)

	# Calculate the Gaussian value corresponding to each element 
	# Gaussian function: exp(- x^2 / (2*sigma^2))
	filter = np.exp((- np.square(arr) / (2 * sigma ** 2)))

	# normalize values in  the filter so that they sum to 1
	# return np.asfarray(filter / sum(filter))
	return np.array(filter) / np.sum(filter)

# Q3
def gauss2d(sigma):
	"This returns a 2D Gaussian filter for a given value of sigma"

	f = gauss1d(sigma)
	# make 2d
	f = f[np.newaxis]
	ft = f.transpose()
	
	return np.asfarray(signal.convolve(f, ft))
	
# Q4.a
def gaussconvolve2d(array, sigma):
	"Applies Gaussian convolution to a 2D array for the given value of sigma, and return the array"

	filter = gauss2d(sigma)
	# return signal.correlate2d(array, filter, 'same')
	return signal.convolve2d(array, filter, 'same')

# Q4.b-c
def grey_gaussian_dog():
	im = Image.open('dog.jpg')

	# convert the image to a greyscale image
	im_grey = im.convert('L')
	# convert to Numpy array for (for subsequent processing)
	im_grey_array = np.asfarray(im_grey)
	gaussconvolve2d(im_grey_array, 3)

	# convert the image to a numpy array (for subsequent processing)
	# convert the result back to a PIL image and save
	im_grey = Image.fromarray(im_grey_array)

	# 4c)
	im.show()
	im_grey.show()

# Part 2
# Q1-Q2
def lohipass_filter(path, sigma, is_high_pass):
	"Filter a image by its color channels separately and turn it into its blurred version"

	im = Image.open(path)
	im_array = np.asfarray(im)

	img_rgb = im.convert('RGB')
	r, g, b = img_rgb.split();
	r, g, b = np.asfarray(r), np.asfarray(g), np.asfarray(b)

	r = gaussconvolve2d(r, sigma)
	g = gaussconvolve2d(g, sigma)
	b = gaussconvolve2d(b, sigma)

	img_rgb_array = np.dstack((r, g, b))

	if is_high_pass: img_rgb_array = np.subtract(im_array, img_rgb_array) + scaler

	img_rgb = Image.fromarray(np.uint8(img_rgb_array))
	img_rgb.show()

	return img_rgb_array

# Q3
def hybrid_filter(low_path, high_path, low_sigma, high_sigma):
	"Convert two images (1 high and 1 low frequency image) into one blurry hybrid image"
	"using lohipass_filter()"

	low = lohipass_filter(low_path, low_sigma, False)
	high = lohipass_filter(high_path, high_sigma, True)
	im = Image.fromarray(np.uint8(np.clip(np.add(low, high - scaler), lower_range, upper_range)))
	im.show()

	return im


