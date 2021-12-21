from PIL import Image
import PIL
import pytesseract
from pytesseract import Output
import numpy as np
import re
import cv2
import spacy
import pdf2image
import os
import glob
import json
import pandas as pd
from pdf2image import convert_from_path


# main function
def ocr_main(img):
    text = pytesseract.image_to_string(img)
    return text


# reading the image

img = cv2.imread('invoice.jpg')


# PREPROCESSING THE IMAGE

# Gray-scaling
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 3)


# Thresholding
def thresholding(image):
    return cv2.threshold(image, 100, 230, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Dilation
def dilate(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# Erosion
def erode(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# Opening erosion followed by dilation
def opening(image):
    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# resizing the image
def resize(image):
    return cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


# skewness correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# Canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# Median blur
def gauss(image):
    return cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)


# Calling Preprocessing functions according to user needs

img = get_grayscale(img)
img = thresholding(img)
img = remove_noise(img)
img = erode(img)
img = dilate(img)
img = opening(img)
img = resize(img)

# Creating a dictionary to store OCR results
d = pytesseract.image_to_data(img, output_type=Output.DICT)
# print(d.keys())


# Creating Bounding Boxes to view
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# OEM CONTROLS TYPE OF Algo used and psm controls page segmentation
custom_oem_psm_config = r'--oem 3 --psm 11'
print(ocr_main(img))

text = pytesseract.image_to_string(img, lang='eng', config=custom_oem_psm_config)
nlp = spacy.load("en_core_web_sm")
sents = nlp(text)
count = 1
data_list = []

# Text Labelling on entities

for sent in sents:
    # print(sent)
    tokens = sent.text.split(" ")
    for i in range(len(tokens)):
        var = tokens[i]
        data = {}

        # print(var)
        if re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', str(var)):
            print("GST Number", str(var))
            data.update({"GST Number": str(var)})
        elif re.search(r'^[a-zA-Z]{5}[0-9]{4}[a-zA-Z]$', re.sub(r'\s+', '', str(var))):
            print("PAN Number", str(var))
            data.update({"PAN Number": str(var)})
        # Using Regular expression to match dates
        elif re.match(r'^(0[1-9]|[12][0-9]|3[01]).(0[1-9]|1[012]).(19|20)\d\d$', str(var)):
            print("Date of Invoice", str(var))
            data.update({"Date of Invoice": str(var)})
        # Simple number for amount
        elif re.match(r'\d{1,6}', str(var)):
            print("Amount", str(var))
            data.update({"Amount": str(var)})

        count = count + 1
        if data != {}:
            data_list.append(data)

# Printing the data dictionary
print(data_list)

# Exporting result to CSV file
df = pd.DataFrame(data_list)
df.to_csv('output.csv')

# Using OpenCV  to Preview Preprocessed image

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()