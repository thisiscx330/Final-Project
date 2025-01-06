#把框出來的文字存在資料夾
#python CCL2.py -i car2.png -o output
from skimage import measure
import numpy as np
import cv2
import argparse
import imutils
import os

# 初始化參數解析器
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-o", "--output", required=True, help="Path to save the character images")
args = vars(ap.parse_args())

# 載入影像並轉換為灰階
image = cv2.imread(args["image"])
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

# 適應性閾值處理
thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 3)

# 顯示處理後的影像
cv2.imshow("License Plate", image)
cv2.imshow("Thresh", thresh)

# 連通區域分析
labels = measure.label(thresh, connectivity=2, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
print(f"[INFO] found {len(np.unique(labels))} blobs")

# 建立輸出資料夾
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

character_count = 0
for (i, label) in enumerate(np.unique(labels)):
    # 忽略背景標籤
    if label == 0:
        print("[INFO] label: 0 (background)")
        continue

    # 創建標籤遮罩
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    # 過濾小區域
    if 500 <= numPixels <= 1500:
        mask = cv2.add(mask, labelMask)

# 從遮罩中提取字符區域
clone = image.copy()
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    # 擬合矩形框
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 裁剪字符並保存
    char_image = image[y:y + h, x:x + w]
    char_path = os.path.join(args["output"], f"char_{character_count}.png")
    cv2.imwrite(char_path, char_image)
    character_count += 1

    # 顯示框選後的影像
    cv2.imshow("Bounding Boxes", clone)

# 顯示檢測結果
cv2.imshow("Large Blobs", mask)
cv2.waitKey(0)
print(f"[INFO] {character_count} characters saved to {args['output']}")
