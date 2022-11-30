
# 椒盐噪声生成器
import cv2 as cv
import numpy as np

def piper_demo(image):
    copy_image = image.copy()
    numb_of_noise = 200000  # number of piper noise point(SNR)
    w, h = copy_image.shape[:2]
    for i in range(numb_of_noise):
        copy_image[np.random.randint(w), np.random.randint(h - 1)] = \
            copy_image[np.random.randint(w-1), np.random.randint(h)] = \
            copy_image[np.random.randint(w-1), np.random.randint(h-1)] = \
            copy_image[np.random.randint(w), np.random.randint(h)] = \
            np.random.randint(10, size=(3)) * 255
    cv.imshow('piper_image', copy_image)
    return copy_image


# def blur_demo(image):
#     mean_blur_image = cv.blur(image, (5, 5))
#     cv.imshow('mean_image', mean_blur_image)
#
#     median_blur_image = cv.medianBlur(image, 5)
#     cv.imshow('media_image', median_blur_image)


src = cv.imread('./data/0001.jpg')
# cv.imshow('Img1', src)

piper_image = piper_demo(src)
cv.imwrite('./noisy/IMG_01Noisy.bmp', piper_image)
# cv.imshow('Fruits', piper_image)
# blur_demo(piper_image)
# cv.imshow('original_image3', piper_image)
cv.waitKey(0)