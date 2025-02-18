
import cv2
import paddleocr
from paddleocr import PaddleOCR, draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = "/Users/sanikabharvirkar/Documents/pprlastshot/match01_paddle_output/frame_0739.jpg"
img = cv2.imread(img_path)
inverted_image = cv2.bitwise_not(img)
cv2.imwrite(img_path, inverted_image)
#img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#scale_factor = 1.0  # Magnify by 2x

# Get new dimensions
#height, width = img.shape[:2]
#new_dimensions = (int(width * scale_factor), int(height * scale_factor))

# Resize the image
#magnified_image = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)
#alpha = 1.5  # Contrast control (1.0-3.0)
#beta = 0     # Brightness control (0-100)

# Apply the transformation
#magnified_image = cv2.convertScaleAbs(magnified_image, alpha=alpha, beta=beta)

#magnified_image = cv2.detailEnhance(magnified_image, sigma_s=5, sigma_r=0.15)
#gray_image = cv2.cvtColor(magnified_image, cv2.COLOR_BGR2GRAY)magnified_image = cv2.equalizeHist(gray_image)

#cv2.imwrite(img_path, magnified_image)

result = ocr.ocr(img_path, cls=True)
print(len(result))
for line in result:
    print(len(line))
    for text in line: 
        print(text[1][0])


