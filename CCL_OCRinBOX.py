#框出車牌後OCR辨識框內的文字

from skimage import measure
import numpy as np
import cv2
import argparse
import imutils
import pytesseract

# 主程式
def process_plate_with_ocr(image_path):
    # 讀取圖片並轉為灰階
    image = cv2.imread(image_path)
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 17, 3)

    # 進行連通元件分析
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8"
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if 500 < numPixels < 1500:  # 篩選合理大小的元件
            mask = cv2.add(mask, labelMask)

    # 找出連通元件的輪廓並排序
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # 按x座標從左到右排序

    plate_number = ""
    for (x, y, w, h) in bounding_boxes:
        # 提取每個框的字符
        char_image = thresh[y:y + h, x:x + w]
        char_image_resized = cv2.resize(char_image, (50, 100))  # 調整大小以統一處理
        char_image_inverted = cv2.bitwise_not(char_image_resized)  # 反轉顏色以適配 OCR

        # 使用 OCR 辨識字符
        recognized_char = pytesseract.image_to_string(
            char_image_inverted, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()
        plate_number += recognized_char

        # 在原圖上繪製框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示最終結果
    print(f"辨識出的車牌號碼: {plate_number}")
    cv2.imshow("Final Result", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    # 替換為你的車牌圖片路徑
    image_path = "car2.png"  # 輸入圖片

    try:
        process_plate_with_ocr(image_path)
    except Exception as e:
        print(f"錯誤: {e}")
