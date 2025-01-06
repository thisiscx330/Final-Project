#OCR辨識強化後的車牌
#python OCR.py
import cv2
from PIL import Image
import sys
import pyocr
import pyocr.builders
import re

# 讀取圖片
image = cv2.imread('afterclear.png')
# 顯示圖片
cv2.imshow('After Clear Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#OCR辨識車牌
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
tool = tools[0]  #取得可用工具

result = tool.image_to_string(
    Image.open('afterclear.png'),
    builder=pyocr.builders.TextBuilder()
)
# 將 ocr 辨識結果優化
txt=result.replace("!","1") # 如果是 ! 字元，更改為字元 1
txt=result.replace("|","1")
real_txt=re.findall(r'[A-Z]+|[\d]+',txt) #只取數字和大寫英文字母

#組合真正的車牌
txt_Plate=""
for char in real_txt:
    txt_Plate += char
print("OCR 辨識結果：", result)
print("優化後辨識結果：",txt_Plate)

cv2.imshow('image', image)     #顯示原始圖形
cv2.moveWindow("image",500,250)#將視窗移到指定位置
key = cv2.waitKey(0)           #按任意鍵結束
cv2.destroyAllWindows()