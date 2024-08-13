import cv2
import numpy as np
import pyocr

from PIL import Image

# cartao japones
img = cv2.imread("cartao.jpg")
# pagina livro
#img = cv2.imread("pagina1.png")
#constituicao
#img = cv2.imread("ap.png")
cv2.imshow('color image', img)
cv2.waitKey(0)

#Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Binarization
ret,th1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
#ret,th1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
cv2.imshow('color image', th1)
cv2.waitKey(0)
cv2.imwrite('c-ph1.png', th1)

#Contour extraction
contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Sort only those with a large area
areas = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10000:
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        areas.append(approx)

cv2.drawContours(img,areas,-1,(0,255,0),3)

cv2.imshow('color image', img)
cv2.waitKey(0)
cv2.imwrite('c-ph2.png', img)

dst = []
pts1 = np.float32(areas[0])
# cartao japones
pts2 = np.float32([[600,300], [600, 0], [0,0], [0,300]])
#pagina livro
#pts2 = np.float32([[0,0], [0,300], [300,300], [300,0]])

M = cv2.getPerspectiveTransform(pts1, pts2)
#cartao japones
dst = cv2.warpPerspective(img, M, (600,300))
#pagina livro
#dst = cv2.warpPerspective(img, M, (300,300))
#constituicao
#dst = cv2.warpPerspective(img, M, (600,300))

cv2.imshow('color image', dst)
cv2.waitKey(0)
cv2.imwrite('c-ph3.png', dst)

tools = pyocr.get_available_tools()
tool = tools[0]
#cartao japones
print(tool.image_to_string(Image.fromarray(dst), lang="jpn"))
#pagina livro
#print(tool.image_to_string(Image.fromarray(dst), lang="eng"))
