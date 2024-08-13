import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyocr
from PIL import Image

# Load the image
img = cv2.imread('ap.png') 
 
# Create a copy of the image
img_copy = np.copy(img)
 
# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the 
# transformation matrix
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
 
plt.imshow(img_copy)


# All points are in format [cols, rows]
pt_A = [346, 654]
pt_D = [1940, 436]
pt_B = [490, 3118]
pt_C = [2860, 2583]


# Here, I have used L2 norm. You can use L1 also.
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([
		                [0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
		                [maxWidth - 1, 0]
		                ])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)

out = cv2.warpPerspective(img,M,(maxWidth, maxHeight))


cv2.imwrite('ap-pj.png', out)

tools = pyocr.get_available_tools()
tool = tools[0]
#cartao japones
print(tool.image_to_string(Image.fromarray(out), lang="eng"))