#強化影像後儲存成afterclear.png並OCR辨識
#python LPR.py
import cv2
import numpy as np
from PIL import Image
import sys
import pyocr
import pyocr.builders
import re

# 影像強化處理
def enhance_image(input_image_path, output_image_path):
    image = cv2.imread(input_image_path)
    print(f"讀取影像 {input_image_path}")
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    #轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # 調整對比度和亮度
    alpha = 2.5
    beta = -50
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 使用 Otsu 閾值法進行二值化
    _, thresh = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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

    # 儲存處理後的圖片
    cv2.imwrite(output_image_path, thresh)
    print(f"影像強化完成 儲存至 {output_image_path}")

    # 顯示強化後的圖片
    enhanced_image = cv2.imread(output_image_path)
    cv2.imshow('afterclear', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# OCR 辨識處理
def perform_ocr(image_path):
    # 初始化 OCR 工具
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]

    # OCR 辨識
    result = tool.image_to_string(
        Image.open(image_path),
        builder=pyocr.builders.TextBuilder()
    )

    # 優化 OCR 結果
    txt = result.replace("!", "1").replace("|", "1")
    real_txt = re.findall(r'[A-Z]+|[\d]+', txt)

    # 組合真正的車牌
    txt_Plate = "".join(real_txt)
    print("OCR 辨識結果：", result)
    print("優化後辨識結果：", txt_Plate)

# 主程式
if __name__ == "__main__":
    input_image = 'MCA9277.jpg'  # 原始圖片名稱
    enhanced_image = 'afterclear.png'  # 儲存強化後圖片的名稱

    # 影像強化
    enhance_image(input_image, enhanced_image)

    # OCR 辨識
    perform_ocr(enhanced_image)
