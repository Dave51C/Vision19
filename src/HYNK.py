import multiprocessing as mp
import cv2
import numpy as np
import time
def image_processing (wkimg):
#    dst = cv2.resize(wkimg,None,fx=0.20,fy=0.20,interpolation=cv2.INTER_AREA)
    dst=wkimg
    wkimg=dst
    jevois.writeText(wkimg, "OINK! OINK! OINK!", 200, 120, jevois.YUYV.White, jevois.Font.Font10x20)
    """
    do all that GRIP stuff
    do all that Target stuff
    """
#    print(STR)
#    time.sleep(2)
#    cv2.imwrite('dst.png',dst)

n=0
fred="fred"
STR="TADA!"
while True:
    inimg = inframe.getCvBGR()
    p     = mp.Process(target=image_processing,args=(inimg,))
    n+=1
    p.start()
    p.join()
    outframe.sendCvBGR(inimg)
