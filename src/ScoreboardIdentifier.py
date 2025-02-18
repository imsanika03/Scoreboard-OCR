import ssl
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
            self.ocr_image_preprocessing(valid_board, save_proprocessed=True)

            text_result = self.ocr.ocr(self.image_path, cls=True)
            return True, text_result
        
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
        
        img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale_factor = 3  # Magnify by 2x

        # Get new dimensions
        height, width = img.shape[:2]
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))

        # Resize the image
        magnified_image = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)

        #crop the noise from top and bottom
        height, width = magnified_image.shape[:2]
        top_crop = height // 10  # 1/8th of the height
        bottom_crop = height - top_crop
        magnified_image = magnified_image[top_crop:bottom_crop, :]

        # invert image since paddleocr trained dataset doesnt include numbers with black background
        inverted_image = cv2.bitwise_not(magnified_image)

        if save_proprocessed: 

            cv2.imwrite(self.image_path, inverted_image)
            return inverted_image

        return magnified_image

    
    def update_image(self, image_path): 
        self.image_path = image_path