import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.linalg import inv
import csv
from scipy.interpolate import RegularGridInterpolator

def readLUT(filename):
    with open(filename, 'rb') as lutfile:
        LUT_table = []
        skip_lines = 3
        datareader = csv.reader(lutfile)
        for i in range(skip_lines):
            datareader.next()

        for row in datareader:
            item = map(float, row[0].split(" "))
            LUT_table.append(item)
        LUT_table = np.array(LUT_table)
        # print LUT_table.shape
        LUT_table = LUT_table.reshape((33, 33, 33, 3))
        return LUT_table

def applyLUT(img, lut):
    img_index = img / 0.030304
    img_index = np.minimum(img_index, 32.0)
    # img_index = img_index.astype(int)
    x = np.linspace(0, 32, 33)
    y = np.linspace(0, 32, 33)
    z = np.linspace(0, 32, 33)
    x_data, y_data, z_data = np.split(lut, 3, axis = 3)
    x_data = np.squeeze(x_data)
    y_data = np.squeeze(y_data)
    z_data = np.squeeze(z_data)
    x_interpolating_function = RegularGridInterpolator((x, y, z), x_data)
    y_interpolating_function = RegularGridInterpolator((x, y, z), y_data)
    z_interpolating_function = RegularGridInterpolator((x, y, z), z_data)
    result_img = np.zeros_like(img_index, dtype=np.float)
    for x in range(0, img_index.shape[0]):
        for y in range(0, img_index.shape[1]):
            select_index = img_index[x, y, :]
            pts = np.array([[select_index[0], select_index[1], select_index[2]]])
            result_img[x, y, 0] = x_interpolating_function(pts)
            result_img[x, y, 1] = y_interpolating_function(pts)
            result_img[x, y, 2] = z_interpolating_function(pts)
            # result_img[x, y, :] = img_index[x, y, :]
    return result_img

def wbmask(m, n, wbmults, align):
    colormask = wbmults[1] * np.ones((m,n), np.float32)
    if align == 'rggb':
        colormask[0::2,0::2] = wbmults[0]
        colormask[1::2,1::2] = wbmults[2]
    elif align == 'bggr':
        colormask[1::2,1::2] = wbmults[0]
        colormask[0::2,0::2] = wbmults[2]
    elif align == 'grbg':
        colormask[0::2,1::2] = wbmults[0]
        colormask[0::2,1::2] = wbmults[2]
    elif align == 'gbrg':
        colormask[1::2,0::2] = wbmults[0]
        colormask[0::2,1::2] = wbmults[2]
    return colormask

def demosaicing(raw_image, align):
    red_channel = raw_image.copy()
    green_channel = raw_image.copy()
    blue_channel = raw_image.copy()
    for x in range(0, raw_image.shape[0]):
        for y in range(0, raw_image.shape[1]):
            if align == 'rggb':
                if x % 2 == 0 or y % 2 == 0:
                    blue_channel[x,y] = 0
                if (x+y) % 2 == 0:
                    green_channel[x, y] = 0
                if x % 2 != 0 or y % 2 != 0:
                    red_channel[x, y] = 0

    green_kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0
    red_blue_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0

    blue_channel = cv2.filter2D(blue_channel, -1, red_blue_kernel)
    green_channel = cv2.filter2D(green_channel, -1, green_kernel)
    red_channel = cv2.filter2D(red_channel, -1, red_blue_kernel)

    return red_channel, green_channel, blue_channel

def apply_cmatrix(img, cmatrix):
    r = cmatrix[0,0] * img[:,:,0] + cmatrix[0, 1] * img[:,:,1] + cmatrix[0,2] * img[:,:,2]
    g = cmatrix[1,0] * img[:,:,0] + cmatrix[1, 1] * img[:,:,1] + cmatrix[1,2] * img[:,:,2]
    b = cmatrix[2,0] * img[:,:,0] + cmatrix[2, 1] * img[:,:,1] + cmatrix[2,2] * img[:,:,2]

    rgb_img = cv2.merge((r, g, b))
    return rgb_img

# raw_image = cv2.imread("./L1004220.tiff",0)
im = Image.open("./nikond7000_iso100.tiff")
raw_image = np.array(im)
align = 'rggb'
LUT_name = "./ArriLogCtoRec709.dat"
# LUT_name = "./DJI_X5_DLOG2sRGB_Improv.cube"
# cv2.imshow("Raw", raw_image)
# black = 44.0
black = 0.0
saturation = 16383.0
print np.amax(raw_image)
lin_bayer = (raw_image - black) / (saturation - black)
lin_bayer = np.minimum(lin_bayer, 1.0)
lin_bayer = np.maximum(lin_bayer, 0.0)
print lin_bayer.shape
wbmults = np.array([1.98, 1, 1.38], np.float32)
m,n = lin_bayer.shape
mask = wbmask(m, n, wbmults, align)
# print mask
balanced_bayer = np.multiply(lin_bayer, mask)

red_channel, green_channel, blue_channel = demosaicing(balanced_bayer, align)

rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)

xyz2cam  = np.array([[8198, -2239, -724],
                     [-4871, 12389, 2798],
                     [-1043, 2050, 7181]], np.float32) #nikon D7000

rgb2cam = xyz2cam * rgb2xyz
# print rgb2cam
rgb2cam = rgb2cam / rgb2cam.sum(axis=1)[:,None]
# print rgb2cam
cam2rgb = inv(rgb2cam)
# print cam2rgb
# rgb_image = cv2.merge((blue_channel, green_channel, red_channel))
rgb_image = cv2.merge((red_channel, green_channel, blue_channel)) #rgb mode
rgb_image = cv2.resize(rgb_image, (750, 500))

lin_srgb = apply_cmatrix(rgb_image, cam2rgb)
lin_srgb = np.maximum(np.minimum(lin_srgb, 1.0), 0.0)
# print lin_srgb
# lin_srgb = applyLUT(lin_srgb, readLUT(LUT_name))
gray_image = np.dot(lin_srgb[:,:,:3], [0.299, 0.587, 0.114])
# gray_image = cv2.cvtColor(lin_srgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
grayscale = 0.25 / gray_image.mean()
# print grayscale
# cv2.imwrite("./wql_gray.tiff", grayscale * gray_image * 255)
bright_srgb = np.minimum(1, lin_srgb * grayscale)
# nl_srgb = applyLUT(bright_srgb, readLUT(LUT_name))
nl_srgb = np.power(bright_srgb, 1/2.0)

r, g, b = cv2.split(nl_srgb)

nl_srgb = cv2.merge((b,g,r))
# nl_srgb = cv2.resize(nl_srgb, (750, 500))
nl_srgb = applyLUT(nl_srgb, readLUT(LUT_name))

cv2.imwrite("./wql_result.jpeg", nl_srgb * 255)
