import libjevois as jevois
import multiprocessing as mp
import cv2
import numpy as np

class JFSM:
    ## Process function with USB output
    inframe = np.zeros([480, 640, 3], dtype=np.uint8)
    def process(self, inframe, outframe):
        N=0
        def image_processing (wkimg):
            #dst=wkimg
            #wkimg=dst
            #jevois.writeText(wkimg, "OINK! OINK! OINK!", 200, 120, jevois.YUYV.White, jevois.Font.Font10x20)
            jevois.LINFO('frame ',N)
            cv2.putText(inimg, "OINK! OINK! OINK!", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
            """
            do all that GRIP stuff
            do all that Target stuff
            """
        inimg  = inframe.getCvBGR()
        N+=1
        p      = mp.Process(target=image_processing,args=(inimg,))
        p.start()
        p.join()
        outimg = inimg
                
        # Write a title:
        #cv2.putText(outimg, "OINK! OINK! OINK!", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
        outframe.sendCvBGR(outimg)
