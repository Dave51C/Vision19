import multiprocessing as mp
import time
import cv2
pipe_r, pipe_w = mp.Pipe()         # create pipe for passing image

def image_processing (comm):
    print("Processing")
    outimg = comm.recv()
    time.sleep(5)
    """
    do all that GRIP stuff
    do all that Target stuff
    """
#    sendCvBRG(outimg)
    dst = cv.resize(outimg,None,fx=0.20,fy=0.20,interpolation=cv.INTER_AREA)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)


#inimg=inframe.getCvBGR()
inimg = cv2.imread('../images/Field/IMG_6339.JPG')
while True:
    print("Spawning")
    p     = mp.Process(target=image_processing,args=(pipe_r,))
    print("Sending")
    pipe_w.send(inimg)
    p.start()
    print("Waiting")
#    inimg = inframe.getCvBGR()
    inimg = cv2.imread('../images/Field/IMG_6339.JPG')
    p.join()
#    inframe.done()
