from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc

scale = 0.75
threshold = 0.55
pyramid = a2.MakePyramid("./faces/judybats.jpg", 20)
template = "./faces/template.jpg"
template_width = 15
t = Image.open(template).resize((template_width, template_width), Image.BICUBIC)

for i, image in enumerate(pyramid):
	arr_nmcorrl = ncc.normxcorr2D(image, t)
	upscale = (1 / scale) ** i # scale to be multiple regards to original image
	#height, width = arr_nmcorrl.shape
	width, height = arr_nmcorrl.shape[::-1]
	print width, height
	#print range(width), range(height)
	# To find every pixel that is above the threshold, and output the peaks scale to the original image
	peaks = []
	for x in range(width):
		for y in range(height):
			if (arr_nmcorrl[y][x] > threshold): peaks.append([x * upscale, y * upscale])
	print peaks



import a2
threshold = 0.571
#pyramids
judybats = a2.MakePyramid("./faces/judybats.jpg", 20)
family = a2.MakePyramid("./faces/family.jpg", 20)
fans = a2.MakePyramid("./faces/fans.jpg", 20)
sports = a2.MakePyramid("./faces/sports.jpg", 20)
students = a2.MakePyramid("./faces/students.jpg", 20)
tree = a2.MakePyramid("./faces/tree.jpg", 20)

template = "./faces/template.jpg"
a2.FindTemplate(judybats, template, threshold)
a2.FindTemplate(family, template, threshold)
a2.FindTemplate(fans, template, threshold)
a2.FindTemplate(sports, template, threshold)
a2.FindTemplate(students, template, threshold)
a2.FindTemplate(tree, template, threshold)