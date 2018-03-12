import numpy as np
import cv2
import copy


def edgedetect(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255  # Some values seem to go above 255. However RGB channels has to be within 0-255
    return sobel


def findSignificantContours(img, edgeImg):
    contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)
    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = edgeImg.size * 5 / 100  # If contour isn't covering 5% of total area of image then it probably is too small
    gcontour = 0
    garea = 0
    for tupl in level1:
        contour = contours[tupl[0]]

        epsilon = 0.10 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 3, True)
        contour = approx

        area = cv2.contourArea(contour)
        if area > garea:
            garea = area
            gcontour = contour

    significant.append([gcontour, garea])
    # Draw the contour on the original image
    #cv2.drawContours(img, [gcontour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    significant.sort(key=lambda x: x[1])
    # print ([x[1] for x in significant]);
    return [x[0] for x in significant]


def get_maske_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Remove noise
    edgeImg = np.max(
        np.array([edgedetect(blurred[:, :, 0]), edgedetect(blurred[:, :, 1]), edgedetect(blurred[:, :, 2])]), axis=0)

    mean = np.mean(edgeImg)
    # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg <= mean] = 0
    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    # Find contours
    significant = findSignificantContours(img, edgeImg_8u)

    # Mask
    mask = edgeImg.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)

    # Finally remove the background
    img[mask] = 0
    return img


def mask_bk_image(bk_image, mask_image, resized):
    for i in range(mask_image.shape[0]):
        for j in range(mask_image.shape[1]):
            if sum(mask_image[i][j]) != 0:
                bk_image[i][j] = resized[i][j]
    return bk_image

def merge_bk_and_matting_product(bk_image, product_image):
    dim = product_image.shape
    bk_image = cv2.resize(bk_image, (dim[1], dim[0]), interpolation=cv2.INTER_AREA)
    for i in range(300):
        for j in range(400):
            if sum(product_image[i][j]) > 3:
                bk_image[i][j] = product_image[i][j]
    return bk_image

def merge_bk_image(bk_image, product_image, directly):
    #if directly == True:
    return merge_bk_and_matting_product(bk_image,product_image)

    #mask_image = get_maske_image(product_image)
    #return mask_bk_image(bk_image, mask_image, product_image)