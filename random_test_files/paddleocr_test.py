
import cv2
from matplotlib import pyplot as plt
import numpy as np
import paddleocr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4', use_space_char=True) # need to run only once to download and load model into memory
img_path = "/Users/sanikabharvirkar/Documents/pprlastshot/random_test_files/inner_box_final.png"
img = cv2.imread(img_path)
# Check if the image was loaded successfully


def make_non_black_pixels_white(image_path, output_path):
    # Open the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask where black pixels remain (thresholding)
    mask = gray < 100  # Adjust threshold if needed

    # Create a white canvas
    output = np.ones_like(image) * 255  # White background

    # Keep only black pixels
    output[mask] = image[mask]
    cv2.imwrite(output_path, output)
    # Return the processed image path


# Example usage:
make_non_black_pixels_white(img_path, "/Users/sanikabharvirkar/Documents/pprlastshot/random_test_files/inner_box_final_b&w.png")

img = cv2.imread("/Users/sanikabharvirkar/Documents/pprlastshot/random_test_files/outer_box_final2.png")
img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
scale_factor = 2  # Magnify by 2x

# Get new dimensions
height, width = img.shape[:2]
new_dimensions = (int(width * scale_factor), int(height * scale_factor))

# Resize the image
magnified_image = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)


# invert image since paddleocr trained dataset doesnt include numbers with black background
inverted_image = cv2.bitwise_not(magnified_image)

plt.imshow(inverted_image)
plt.axis('off')
plt.show()

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

result = ocr.ocr(inverted_image, cls=True)

for line in result:

    for text in line: 
        print(text[1][0])


