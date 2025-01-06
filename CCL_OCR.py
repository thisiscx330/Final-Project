#框出文字並用OCR辨識
#python CCL_OCR.py
from skimage import measure
import numpy as np
import cv2
from PIL import Image
import pyocr
import pyocr.builders
import re
import imutils

def ccl_and_display(image_path):
    # 讀取圖片並處理亮度分量
    print(f"讀取影像 {image_path}")
    image = cv2.imread(image_path)
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

    # 自適應閾值處理
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 17, 3)

    # 顯示原始圖片和閾值處理結果
    cv2.imshow("License Plate", image)
    cv2.imshow("Thresh", thresh)

    # 連通元件分析
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    print("[INFO] Found {} blobs".format(len(np.unique(labels))))

    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if 500 < numPixels < 1500:
            mask = cv2.add(mask, labelMask)

    # 顯示篩選後的元件
    cv2.imshow("Large Blobs", mask)

    # 繪製框選結果
    clone = image.copy()
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Bounding Boxes", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ocr_recognition(image_path):
    # 使用 pyocr 進行 OCR 辨識
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        return None

    tool = tools[0]
    result = tool.image_to_string(
        Image.open(image_path),
        builder=pyocr.builders.TextBuilder()
    )

    # 優化 OCR 結果
    result = result.replace("!", "1").replace("|", "1")
    real_txt = re.findall(r'[A-Z]+|[\d]+', result)
    txt_Plate = "".join(real_txt)

    # 輸出結果
    print("OCR 辨識結果：", result)
    print("優化後辨識結果：", txt_Plate)

def main():
    # 讀取的圖片
    image_path = "car.png"

    # 執行連通元件分析與顯示
    ccl_and_display(image_path)

    # 執行 OCR 辨識
    ocr_recognition(image_path)

if __name__ == "__main__":
    main()
