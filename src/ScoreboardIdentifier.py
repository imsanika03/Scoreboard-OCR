import ssl

from matplotlib import pyplot as plt
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch
from PIL import Image
 
import paddleocr
from paddleocr import PaddleOCR, draw_ocr

import sys

class TextIdentifier(): 

    def __init__(self, image_path=None):
       
       self.ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

       self.image_path = image_path
       
       # Model
       model = torch.hub.load('ultralytics/yolov5', 'custom', path='scoreboard_model.pt')
       model.to(torch.device("cpu")).eval()    
       self.model = model 
    
    ## returns text in image with easyocr api 
    def call_ocr(self): 

        valid_board = self.identify_and_crop_score()

        # yolo model found a scoreboard in frame
        if valid_board is not None: 

            # pre process image for better ocr results 
            ok, full_score_box = self.ocr_image_preprocessing(valid_board, save_proprocessed=False)
      
            if not ok: 
                return False, "Full box not found"
            
            full_ocr = self.ocr.ocr(full_score_box, cls=True)
            
            return True, full_ocr
        
        else: 
            return False, "Board not found, skip frame"
   
    ## crops image using yolo model
    def identify_and_crop_score(self): 

        # Model Inference
        results = self.model(self.image_path)  

        # Assuming 'results' is the tensor with predictions
        # Extract the bounding box coordinates for the first detection

        # only run if score board is found in a frame
        if not results.pandas().xyxy[0].empty: 
            
            xmin, ymin, xmax, ymax = results.pandas().xyxy[0].iloc[0][['xmin', 'ymin', 'xmax', 'ymax']]

            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

            # Load the image (replace 'image_path' with your image's file path)
            image = cv2.imread(self.image_path)

            # Crop the image using the bounding box
            cropped_image = image[ymin:ymax, xmin:xmax+2]

            return cropped_image
        
        else: 

            return None
        
    def ocr_image_preprocessing(self, img, save_proprocessed=False): 

        if save_proprocessed:
            cv2.imwrite(self.image_path + "cropped.png", img)

        full_score_box = crop_full_score_box(img)

        if full_score_box is None or full_score_box.size == 0: 
            return False, None
        

        full_score_box_outer = preprocess_outer_box(full_score_box, inner_box=False)

        if save_proprocessed: 

            cv2.imwrite(self.image_path + "full_score_box.png", full_score_box_outer)
        
        return True, full_score_box_outer


    
    def update_image(self, image_path): 
        self.image_path = image_path

def crop_full_score_box(frame): 
    
    original_image = frame.copy()
    image = frame.copy()


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]


    # final bounding box
    if bounding_boxes:

        leftmost = min(bounding_boxes, key=lambda b: b[0])[0]
        rightmost = max(bounding_boxes, key=lambda b: b[0] + b[2])[0] + max(bounding_boxes, key=lambda b: b[0] + b[2])[2]
        top = min(bounding_boxes, key=lambda b: b[1])[1]
        bottom = max(bounding_boxes, key=lambda b: b[1] + b[3])[1] + max(bounding_boxes, key=lambda b: b[1] + b[3])[3]
        
 
        padding = 3  # padding to exclude bounding box lines
        leftmost += padding
        rightmost -= padding
        top += padding
        bottom -= padding
        
        cropped = original_image[top:bottom, leftmost:rightmost]
        return cropped

def crop_inner_score_box(frame): 
    original_image = frame.copy()
    image = original_image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Determine the final bounding box
    if bounding_boxes:

        leftmost_box = bounding_boxes[0]
        rightmost_box = bounding_boxes[0]
        top = bounding_boxes[0][1]
        bottom = bounding_boxes[0][1] + bounding_boxes[0][3]
        
        for box in bounding_boxes:
            x, y, w, h = box
            if x < leftmost_box[0]:
                leftmost_box = box
            if x + w > rightmost_box[0] + rightmost_box[2]:
                rightmost_box = box
            if y < top:
                top = y
            if y + h > bottom:
                bottom = y + h
        
        left_x = leftmost_box[0] + leftmost_box[2]  # right line of left box
        right_x = rightmost_box[0]  # left line of right box
        

        cropped = original_image[top:bottom, left_x:right_x]
        return cropped

def preprocess_inner_box(frame): 

    frame = make_non_black_pixels_white(frame)

    img = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_factor = 2  # Magnify by 2x

    height, width = img.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))

    magnified_image = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)
    return magnified_image

def preprocess_outer_box(frame, inner_box=False): 
    
    img = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_factor = 2  # Magnify by 2x

    height, width = img.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))

    magnified_image = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)

    if inner_box: 

        _, width = magnified_image.shape[:2]

        # Compute x-coordinates for inner score box
        x1 = int((8.1 / 10.3) * width)
        x2 = int((9.3 / 10.3) * width)

        magnified_image[:, x1:x2] = make_non_black_pixels_white(magnified_image[:, x1:x2])
    
        magnified_image[:, 0:x1] = cv2.bitwise_not(magnified_image[:, 0:x1])
        magnified_image[:, x2:] = cv2.bitwise_not(magnified_image[:, x2:])
    
        return magnified_image
    else: 
        inverted_image = cv2.bitwise_not(magnified_image)
        return inverted_image

def make_non_black_pixels_white(frame):

    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = gray < 100  # grey thresh

    output = np.ones_like(image) * 255  # White background

    # only black pixels
    output[mask] = image[mask]
    return output

def crop_outer_score_boxes(frame): 
    
    original_image = frame.copy()
    image = original_image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    if bounding_boxes:
        leftmost = min(bounding_boxes, key=lambda b: b[0])[0]
        rightmost = max(bounding_boxes, key=lambda b: b[0] + b[2])[0] + max(bounding_boxes, key=lambda b: b[0] + b[2])[2]
        top = min(bounding_boxes, key=lambda b: b[1])[1]
        bottom = max(bounding_boxes, key=lambda b: b[1] + b[3])[1] + max(bounding_boxes, key=lambda b: b[1] + b[3])[3]
        

        # find inner box: 
        leftmost_box = bounding_boxes[0]
        rightmost_box = bounding_boxes[0]
        top_line = bounding_boxes[0][1]
        bottom_line = bounding_boxes[0][1] + bounding_boxes[0][3]
        
        for box in bounding_boxes:
            x, y, w, h = box
            if x < leftmost_box[0]:
                leftmost_box = box
            if x + w > rightmost_box[0] + rightmost_box[2]:
                rightmost_box = box
            if y < top:
                top = y
            if y + h > bottom:
                bottom = y + h
        
        left_x = leftmost_box[0] + leftmost_box[2]  
        right_x = rightmost_box[0] 

        if original_image.shape[2] == 4:  
            original_image[top:bottom, left_x:right_x] = [0, 0, 0, 255]
        elif original_image.shape[2] == 3:  
            original_image[top:bottom, left_x:right_x] = [0, 0, 0] 
    
        # remove extra noise
        padding = 3  
        leftmost += padding
        rightmost -= padding
        top += padding
        bottom -= padding
        
        cropped = original_image[top:bottom, leftmost:rightmost]
        return cropped