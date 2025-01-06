#用OCR辨識指定資料夾的影像
#python OCR_folder.py -f output
import os
import cv2
from pytesseract import pytesseract

def configure_tesseract(tesseract_path=None):
    if tesseract_path:
        pytesseract.tesseract_cmd = tesseract_path

def recognize_characters(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    recognized_characters = []

    for image_name in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Use Tesseract OCR to recognize the character
        char = pytesseract.image_to_string(image, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        char = char.strip()  # Remove any whitespace or newlines
        recognized_characters.append(char)

    # Combine recognized characters into a license plate string
    license_plate = ''.join(recognized_characters)
    return license_plate

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, help="Path to the folder containing character images")
    parser.add_argument("-t", "--tesseract", required=False, help="Path to the Tesseract executable (if not in PATH)")
    args = parser.parse_args()

    # Configure Tesseract if a custom path is provided
    if args.tesseract:
        configure_tesseract(args.tesseract)

    # Perform OCR on the images
    license_plate = recognize_characters(args.folder)
    print(f"Recognized License Plate: {license_plate}")
