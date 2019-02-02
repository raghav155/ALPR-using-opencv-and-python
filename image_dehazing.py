# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 13:09:25 2019

@author: Raghav
"""

### References
#
## Single Image Haze Removal Using Dark Channel Prior
# https://doi.org/10.1109/TPAMI.2010.168
#
## Guided Image Filtering
# https://doi.org/10.1109/TPAMI.2012.213
#
###

import math
import numpy as np
import cv2 as cv

L = 256


def get_dark_channel(img, *, size):
    """Get dark channel for an image.

    @param img: The source image.

    @param size: Patch size.

    @return The dark channel of the image.
    """
    
    minch = np.amin(img, axis=2)
    box = cv.getStructuringElement(cv.MORPH_RECT, (size // 2, size // 2))
    return cv.erode(minch, box)


def get_atmospheric_light(img, *, size, percent):
    """Estimate atmospheric light for an image.

    @param img: the source image.

    @param size: Patch size for calculating the dark channel.

    @param percent: Percentage of brightest pixels in the dark channel
    considered for the estimation.

    @return The estimated atmospheric light.
    
    """
    
    m, n, _ = img.shape
    print((m,n,_))

    flat_img = img.reshape(m * n, 3)
    print(flat_img)
    flat_dark = get_dark_channel(img, size=size).ravel()
    percent = 0.1
    count = math.ceil(m * n * percent / 100)
    

    indices = np.argpartition(flat_dark, -count)[:-count]


    return np.amax(np.take(flat_img, indices, axis=0), axis=0)
    


def get_transmission(img, atmosphere, *, size, omega, radius, epsilon):
    """Estimate transmission map of an image.

    @param img: The source image.

    @param atmosphere: The atmospheric light for the image.

    @param omega: Factor to preserve minor amounts of haze [1].

    @param radius: (default: 40) Radius for the guided filter [2].

    @param epsilon: (default: 0.001) Epsilon for the guided filter [2].

    @return The transmission map for the source image.
    """
    division = np.float64(img) / np.float64(atmosphere)
    raw = (1 - omega * get_dark_channel(division, size=size)).astype(np.float32)
    return cv.ximgproc.guidedFilter(img, raw, radius, epsilon)


def get_scene_radiance(img,
                       *,
                       size=15,
                       omega=0.95,
                       trans_lb=0.1,
                       percent=0.1,
                       radius=40,
                       epsilon=0.001):
    """Get recovered scene radiance for a hazy image.

    @param img: The source image to be dehazed.

    @param omega: (default: 0.95) Factor to preserve minor amounts of haze [1].

    @param trans_lb: (default: 0.1) Lower bound for transmission [1].

    @param size: (default: 15) Patch size for filtering etc [1].

    @param percent: (default: 0.1) Percentage of pixels chosen to compute atmospheric light [1].

    @param radius: (default: 40) Radius for the guided filter [2].

    @param epsilon: (default: 0.001) Epsilon for the guided filter [2].

    @return The final dehazed image.
    """
    atmosphere = get_atmospheric_light(img, size=size, percent=percent)
    trans = get_transmission(img, atmosphere, size=size, omega=omega, radius=radius, epsilon=epsilon)
    clamped = np.clip(trans, trans_lb, omega)[:, :, None]
    img = np.float64(img)
    return np.uint8(((img - atmosphere) / clamped + atmosphere).clip(0, L - 1))
    


def process_imgdir(imagePath):
    image = cv.imread(imagePath)
    cv.imshow('Image', image)
    dehazed = get_scene_radiance(image)
    side_by_side = np.concatenate((image, dehazed), axis=1)
    cv.imwrite('./dehazed.jpg', side_by_side)


process_imgdir('./hazed.jpg')
