#用文字圖片(photo資料夾)當樣本辨識車牌號碼
#python usephoto.py
import cv2
import os
import numpy as np

def load_templates(template_folder):
    templates = {}
    for file_name in os.listdir(template_folder):
        if file_name.endswith('.jpg'):
            char = os.path.splitext(file_name)[0]  # 檔名即字元
            path = os.path.join(template_folder, file_name)
            templates[char] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return templates


def match_character(image, templates):
    best_match = None
    best_score = -1
    for char, template in templates.items():
        # 模板匹配
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_match = char
    return best_match


def process_plate(plate_image_path, template_folder):
    # 載入模板
    templates = load_templates(template_folder)

    # 讀取車牌圖片
    image = cv2.imread(plate_image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)  # 顯示原始圖形
    cv2.moveWindow("image", 500, 200)  # 將視窗移到指定位置
    key = cv2.waitKey(0)  # 按任意鍵結束
    cv2.destroyAllWindows()

    # 進行二值化
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 尋找字元輪廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20 and w > 10:  # 篩選出可能的字元
            char_image = binary_image[y:y + h, x:x + w]
            char_images.append((x, char_image))  # 儲存字元與其 x 座標（排序用）

    # 按照 x 座標排序字元
    char_images = sorted(char_images, key=lambda x: x[0])

    # 逐一辨識字元
    plate_number = ""
    for _, char_image in char_images:
        # 調整到與模板大小一致
        char_image_resized = cv2.resize(char_image, (templates['A'].shape[1], templates['A'].shape[0]))
        matched_char = match_character(char_image_resized, templates)
        if matched_char:
            plate_number += matched_char

    return plate_number


if __name__ == "__main__":
    # 車牌圖片路徑
    plate_image_path = 'char_A.jpg'  # 車牌圖片
    # 字元模板資料夾路徑
    template_folder = 'photo'  # 放置 0.jpg ~ Z.jpg 的資料夾

    # 辨識車牌
    plate_number = process_plate(plate_image_path, template_folder)
    print(f"車牌號碼: {plate_number}")

