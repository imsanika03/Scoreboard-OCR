
import cv2
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np
import easyocr 

img_path = "/Users/sanikabharvirkar/Documents/pprlastshot/cropped_image.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # Copy white pixels to the output image
img = cv2.bitwise_and(img, img, mask=mask)
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)
img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

text_result = pytesseract.image_to_string(Image.open("/Users/sanikabharvirkar/Documents/pprlastshot/test2.png"))
print(text_result)





#img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def white_pixels_only(img):
    # Read the image
    # Read the image

    # Convert to grayscale to simplify the mask creation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a mask where white pixels are 255
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Copy white pixels to the output image
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("test.jpg", result)
    return result

def extract_contours(img):
    # Read the image


    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(gray, 130, 200)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw contours
    contour_image = np.zeros_like(img)

    # Draw contours on the blank image
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

    # Save the resulting image
    cv2.imwrite("contour_img.png", contour_image)
    return contour_image

img = extract_contours(img)

results = pytesseract.image_to_data(img, output_type=Output.DICT)#OCR'ing the image
for i in range(0, len(results["text"])):
    
    #bounding box coordinates
    x = results['left'][i]
    y = results['top'][i]
    w = results['width'][i]
    h = results['height'][i]
    
    #Extract the text
    text = results["text"][i]
    conf = int(results["conf"][i])# Extracting the confidence
    
    #filtering the confidence
    if conf > 40:
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")
        
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

cv2.imwrite("test.jpg", img)

