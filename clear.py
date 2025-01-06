#強化影像
#python clear.py
import cv2
import numpy as np

def area(row, col):
    global nn
    if bg[row][col] != 255:
        return
    bg[row][col] = lifearea
    if col > 1 and bg[row][col - 1] == 255:
        nn += 1
        area(row, col - 1)
    if col < w - 1 and bg[row][col + 1] == 255:
        nn += 1
        area(row, col + 1)
    if row > 1 and bg[row - 1][col] == 255:
        nn += 1
        area(row - 1, col)
    if row < h - 1 and bg[row + 1][col] == 255:
        nn += 1
        area(row + 1, col)

# 讀取影像
image = cv2.imread('MCA9277.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)

# 調整對比度和亮度
alpha = 2.5
beta = -50
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
cv2.imshow('Adjusted Contrast & Brightness', adjusted)
cv2.waitKey(0)

# 使用Otsu閾值法進行二值化
_, thresh = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)

# 去除雜點
for i in range(len(thresh)):
    for j in range(len(thresh[i])):
        if thresh[i][j] == 255:
            count = 0
            for k in range(-2, 3):
                for l in range(-2, 3):
                    try:
                        if thresh[i + k][j + l] == 255:
                            count += 1
                    except IndexError:
                        pass
            if count <= 10:
                thresh[i][j] = 0

cv2.imshow('Noise Reduced Image', thresh)
cv2.imwrite('afterclear.png', thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()
