import cv2 as cv
import numpy as np

GREEN_MIN = np.array([55,175,175],np.uint8)
GREEN_MAX = np.array([64,255,255],np.uint8)

img    = cv.imread   ('../images/Field/Small.png')
imghsv = cv.cvtColor (img, cv.COLOR_BGR2HSV)
cv.imshow            ('PNG',img)
cv.imshow            ('HSV',imghsv)
imghsv[:,:,2] -= 20
img    = cv.cvtColor (imghsv, cv.COLOR_HSV2BGR)
cv.imshow            ('Darker?',img)

cv.waitKey(0)
cv.destroyAllWindows()
