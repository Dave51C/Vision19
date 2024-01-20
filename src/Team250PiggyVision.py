import libjevois as jevois
import cv2
import multiprocessing
import numpy as np
import time
import math
import time

class Team250PiggyVision:
    # ###################################################################################################
    ## Constructor--when creating a new object, constructor is called as first step to initialize things for that object
    
    def __init__(self):
        # Create a timer to measure how fast frames are processed
        self.jevois_timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)

        self.__blur_radius = 4.504504504504505

        self.blur_output = None

        self.__cv_extractchannel_src = self.blur_output
        self.__cv_extractchannel_channel = 1.0

        self.cv_extractchannel_output = None

        self.__cv_threshold_src = self.cv_extractchannel_output
        self.__cv_threshold_thresh = 30.0
        self.__cv_threshold_maxval = 255.0
        self.__cv_threshold_type = cv2.THRESH_BINARY

        self.cv_threshold_output = None

        self.__mask_input = self.blur_output
        self.__mask_mask = self.cv_threshold_output

        self.mask_output = None

        self.__normalize_input = self.mask_output
        self.__normalize_type = cv2.NORM_MINMAX
        self.__normalize_alpha = 0.0
        self.__normalize_beta = 255.0

        self.normalize_output = None

        self.__hsv_threshold_input = self.normalize_output
        self.__hsv_threshold_hue = [0.0, 49.07338796517677]
        self.__hsv_threshold_saturation = [0.0, 0.454545996405874]
        self.__hsv_threshold_value = [41.276978417266186, 255.0]

        self.hsv_threshold_output = None

        self.__cv_erode_src = self.hsv_threshold_output
        self.__cv_erode_kernel = None
        self.__cv_erode_anchor = (-1, -1)
        self.__cv_erode_iterations = 2.0
        self.__cv_erode_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_bordervalue = (-1)

        self.cv_erode_output = None

        self.__cv_dilate_src = self.cv_erode_output
        self.__cv_dilate_kernel = None
        self.__cv_dilate_anchor = (-1, -1)
        self.__cv_dilate_iterations = 1.0
        self.__cv_dilate_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_bordervalue = (-1)

        self.cv_dilate_output = None

        self.__find_contours_input = self.cv_dilate_output
        self.__find_contours_external_only = False

        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 25.0
        self.__filter_contours_min_perimeter = 0.0
        self.__filter_contours_min_width = 0.0
        self.__filter_contours_max_width = 1000.0
        self.__filter_contours_min_height = 0.0
        self.__filter_contours_max_height = 1000.0
        self.__filter_contours_solidity = [0, 100]
        self.__filter_contours_max_vertices = 1000000.0
        self.__filter_contours_min_vertices = 0.0
        self.__filter_contours_min_ratio = .3
        self.__filter_contours_max_ratio = .7

        self.filter_contours_output = None
        
    # ############################################ ENTERING THE TWILIGHT ZONE#######################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
#--------------------------The following is simply importing and processing our grip pipeline functions----------------------#
        headless = False

# COPY TO HEADLESS FROM HERE      
        debug = True
        
        start = time.time()
        
        # Constantly look and acquire colormin_perimeter = 0.0
        self.__filter_contours_min_width = 0.0
        self.__filter_contours_max_width = 1000.0
        self.__filter_contours_min_height = 0.0
        self.__filter_contours_max_height = 1000.0
        self.__filter_contours_solidity = [0, 100]
        self.__filter_contours_max_vertices = 1000000.0
        self.__filter_contours_min_vertices = 0.0
        self.__filter_contours_min_ratio = .3
        self.__filter_contours_max_ratio = .7

        self.filter_contours_output = None

        #camera images 

        source0 = inimg = inframe.getCvBGR()
        outimg = inimg 
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.jevois_timer.start()

        ## BEGIN GRIP CODE

        # Runs the OpenCV pipeline and sets all outputs to new values.

        self.__blur_input = source0
        (self.blur_output) = self.__blur(self.__blur_input, "Gaussian_Blur", self.__blur_radius)

        # Step CV_extractChannel0:
        self.__cv_extractchannel_src = self.blur_output
        (self.cv_extractchannel_output) = self.__cv_extractchannel(self.__cv_extractchannel_src, self.__cv_extractchannel_channel)

        # Step CV_Threshold0:
        self.__cv_threshold_src = self.cv_extractchannel_output
        (self.cv_threshold_output) = self.__cv_threshold(self.__cv_threshold_src, self.__cv_threshold_thresh, self.__cv_threshold_maxval, self.__cv_threshold_type)

        # Step Mask0:
        self.__mask_input = self.blur_output
        self.__mask_mask = self.cv_threshold_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)

        # Step Normalize0:
        self.__normalize_input = self.mask_output
        (self.normalize_output) = self.__normalize(self.__normalize_input, self.__normalize_type, self.__normalize_alpha, self.__normalize_beta)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.normalize_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_erode0:
        self.__cv_erode_src = self.hsv_threshold_output
        (self.cv_erode_output) = self.__cv_erode(self.__cv_erode_src, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)

        # Step CV_dilate0:
        self.__cv_dilate_src = self.cv_erode_output
        (self.cv_dilate_output) = self.__cv_dilate(self.__cv_dilate_src, self.__cv_dilate_kernel, self.__cv_dilate_anchor, self.__cv_dilate_iterations, self.__cv_dilate_bordertype, self.__cv_dilate_bordervalue)

        # Step Find_Contours0:
        self.__find_contours_input = self.cv_dilate_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)

        ### End Grip Code
       
        ### Start Non-Grip Common/Default Code

        def sort_contours(cnts, method = "left-to-right"):
            
            jevois.LINFO("In sort contours!" + str(len(cnts)))
                            
            reverse = False
            i=0
        
            if method == "right-to-left" or method == "bottom-to-top":
                reverse = True
            
            if method == "top-to-bottom" or method == "bottom-to-top":
                i=i
            
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
        
            return (cnts, boundingBoxes)    

        def draw_contour(image, c, i):
            M = cv2.moments(c)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
        
            cv2.putText(image, "#{}".format(i+1),(cX-20,cY),font, font_size, (255,255,255), 2)
        
            return image
             
        ### Start Additional Non-Grip Custom Code
          
# Constants and Variables---The following is all of our constants and variables we use in our formulas later in the code. :D
        b = 61
        g = 260
        r = 148
        direction = "hatch"
        font = 1
        font_size = 1  
        w, h = 2, 2;
        contour_keeper = [[0 for x in range(w)] for y in range(h)] 
        tape_keeper = [] 
        cx = 0
        cy = 0
        closest_dock = -99.99
        VERTICAL_FOV = 52
        VERTICAL_PXL_PER_DEGREE = 9.23      # 480/52
        HORIZONTAL_PXL_PER_DEGREE = 9.85    # 640/65
        HORIZON_Y = 500
        camera_distance_from_front_bumper = 5
        #SETTING VALUES FOR HATCH VS CARGO :D
        if direction == "hatch":
            CAMERA_X_OFFSET = 5.5
            CAMERA_HEIGHT = 9.0 #really 10.0 but decreased to increase distance
            TARGET_HEIGHT = 31.5
            CALIBRATION_FACTOR = 0
            DEGREES_OFFSET = -21.5
            VERTICAL_DEGREES_OFFSET = 30.0
        else:
            CAMERA_X_OFFSET = 5.5
            CAMERA_HEIGHT = 46.0
            TARGET_HEIGHT = 31.5
            CALIBRATION_FACTOR = -30
            DEGREES_OFFSET = 7
                
        # Draws all contours on original image in red
        cv2.drawContours(outimg, self.filter_contours_output, -1, (0, 0, 250), 1)
        
        # Draw screen grid #This is the code that sets up a grip pattern on the display screen. Green ticks will be placed at every 10 pixels while red ticks will be placed every 100.
        if headless == False and debug == True:
            # X axis grid
            for pixel in range(100,640, 100):
                cv2.putText(outimg, "||", (pixel, 460), font, font_size, (0, 0, 255), 1, cv2.LINE_AA)
            for pixel in range(20, 640, 10):
                cv2.putText(outimg, "|", (pixel, 460), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            for pixel in range(20, 640, 10):
                cv2.putText(outimg, str(round((pixel%100)*.1)), (pixel, 445), font, font_size, (b, g, r), 1, cv2.LINE_AA)

            # Y axis grid        
            for pixel in range(100, 480, 100):
                cv2.putText(outimg, "=", (1,pixel), font, font_size, (0, 0, 255), 1, cv2.LINE_AA)
            for pixel in range(20, 480, 10):
                cv2.putText(outimg, "-", (1,pixel), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            for pixel in range(20, 480, 10):
                cv2.putText(outimg, str(round((pixel%100)*.1)), (15,pixel), font, font_size, (b, g, r), 1, cv2.LINE_AA)
       
        # Gets number of contours
        NumOfContours = len(self.filter_contours_output)
        #if headless == False: cv2.putText(outimg, "Contours Found: " + str(NumOfContours), (3, 450), font, font_size, (b, g, r), 1, cv2.LINE_AA)
        
        # Sorts contours from left to right
        #newContours = sortByArea(self.filter_contours_output)  # Sorts contours by area
        newContours = sorted(self.filter_contours_output, key=lambda ctr: cv2.boundingRect(ctr)[0])
                               
        #Organize the contours we found
        jevois.LINFO("About to run through contours: " + str(NumOfContours) + " to see if the are tapes")
        if NumOfContours != 0:                   # If there are things to analyze in the image...
            for i in range (NumOfContours):      # Loop through everything found...
                cnt = newContours[i]
                
                rect = cv2.minAreaRect(cnt)
                
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                x,y,theta = cv2.minAreaRect(cnt)

                jevois.LINFO("Contour: " + str(i))
                jevois.LINFO("=================")
                jevois.LINFO("x[0] or CX = " + str(x[0]))
                jevois.LINFO("x[1] or CY = " + str(x[1]))
                jevois.LINFO("y[0] or W = " + str(y[0]))
                jevois.LINFO("y[1] or H = " + str(y[1]))
                jevois.LINFO("angle = " + str(theta))
                                                
                cv2.drawContours(outimg,[box],0,(0,255,0),2)
                if headless == False and debug == True: cv2.putText(outimg, "C" + str(i) + " at " + str(round(theta)) + "'", (round(x[0]-30),round(x[1]+80)), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                #cv2.Point2f rect_points[4]
                
                #x,y,w,h = cv2.boundingRect(cnt) # Get the stats of the contour including width and height
                #cv2.rectangle(outimg,(x,y),(x+w,y+h),(0,255,0),2)  #this draws the bounding rectangle in our image :)
                #contour_keeper[i] = x 
                cnt_area = cv2.contourArea(cnt)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                p = cv2.approxPolyDP(hull, -1, 1)
        
                # Determine if something is a piece of tape correctly angled or not
                #Theta is the angle of the rectangle returned by the minAreaRect method imported above :D
                if (theta > -30 and theta < -4) and theta != 0.0:
                    tape_keeper.insert(i,[x,y,"L"])
                elif (theta > -85 and theta < -60) and theta != 0.0:
                    tape_keeper.insert(i,[x,y,"R"])
   
            # Now we only have angled good tapes, so we loop through and identify docks
            if len(tape_keeper) != 0:
                 jevois.LINFO("Looping through " + str(len(tape_keeper)) + " found tapes and determining docks.")
                 if headless == False and debug == True: cv2.putText(outimg, "Tracking " + str(len(tape_keeper)) + " tapes!",(25,60), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                 left_side = False      # flag indicating just saw a left (right leaning) piece of tape
                 right_side = False     # flag indicating just saw a right (left leaning) piece of tape
                 target_found = False   # flag indicating we found a left and right pair - a port!
                 target_number = 0      # number of pairs found

                 t = len(tape_keeper)   # t gets the number of pieces of tape.                 
                 
                 # Loop through the number of found pairs of tape
                 for ctr in range(t):
                     jevois.LINFO("Looking at a " + tape_keeper[ctr][2])

                     # If it's a left side, indicate it's point right...
                     if str(tape_keeper[ctr][2]) == "R":
                        left_side = True
                     # If it's a right side, indicate it's pointing left...
                     elif str(tape_keeper[ctr][2]) == "L" and left_side == True:
                        target_found = True
                        target_number = target_number + 1
                        left_side = False
                        right_side = False
                        jevois.LINFO("Found a full target!!!!")
                        
                        # Save the center X and center Y for this particular matching pair
                        cx = round(tape_keeper[ctr-1][0][0] + (.5 * (tape_keeper[ctr][0][0]-tape_keeper[ctr-1][0][0]))) + round(CAMERA_X_OFFSET)
                        #cy = round((tape_keeper[ctr][0][1]+tape_keeper[ctr-1][0][1])/2)
                        
                        # THIS IS WHERE WE SET THE X AND Y VALUES OF THE TARGET'S TAPES :D
                        left_tape_x = round(tape_keeper[ctr-1][0][0]+tape_keeper[ctr-1][1][1]) #The value returned as the x coordinate is the upper left hand corner, so we must add the width of the min area rectangle.
                        left_tape_y = round(tape_keeper[ctr-1][0][1])+round(tape_keeper[ctr-1][1][1])/2
                        right_tape_x = round(tape_keeper[ctr][0][0])-33
                        right_tape_y = round(tape_keeper[ctr][0][1])+round(tape_keeper[ctr-1][1][1])/2
                        cy = round((right_tape_y+left_tape_y)/2) #This takes the average of the y values to find the center y value of the target :D
                        
            #--------------------------------------------HORIZONTAL ANGLES--------------------------------------------#
                        #Calculates the horizontal degrees to target using the difference between center x and the center of the image.
                        degrees_to_target = (cx - 320)/HORIZONTAL_PXL_PER_DEGREE + DEGREES_OFFSET
                        #The following finds the horizontal angles to each tapes by subtracting the x values of each and finding the distance in pixels to the center of the image. 
                        #The 9 is the number of pixels per degree in the FOV. 
                        
                        angle_to_left_tape = (left_tape_x - 320)/HORIZONTAL_PXL_PER_DEGREE + DEGREES_OFFSET
                        angle_to_right_tape = (right_tape_x - 320)/HORIZONTAL_PXL_PER_DEGREE + DEGREES_OFFSET
                        
                        if abs(degrees_to_target) < abs(closest_dock):
                            closest_dock = degrees_to_target
                            closest_dock_cx = cx
                            closest_dock_cy = cy
                           
#-------------------Same thing as above but consolidated into one loop for calculating and displaying distance and vertical angle-----------#
            if closest_dock != -99.99:

           #---------------------------------------------VERTICAL ANGLES------------------------------------------------#
                        if headless == False and debug == True: cv2.putText(outimg, "cy = " + str(cy) + " horizon_y " + str(HORIZON_Y),(25,90), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        if headless == False and debug == True: cv2.line(outimg,(0,HORIZON_Y),(640,HORIZON_Y),(0,0,255),thickness=1,lineType=8,shift=0)
                        
                        # Left Tape angle
                        #The following finds the vertical angle of the target to the robot by taking the y coordinate of the tape and subtracting 405 pixels.
                        #the horizon_y = 405 is simply taking into account the camera's tilt (it's calibrated not an exact value) :D
                        #vertical_angle = ((abs((cy-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)/ 2) 

                        # Right Tape--This does the same thing but for right tapes the t-1 is to make sure you're not at the end
                        #if ctr != t - 1: vertical_angle = ((abs((tape_keeper[ctr+1][0][1]-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)/ 2) 
                        
                        #These two lines find the vertical angles of the right and left tapes of the target the same way as finding the vertical angle for turning. :D
                        left_vert = (abs(left_tape_y - HORIZON_Y) / VERTICAL_PXL_PER_DEGREE) + math.radians(VERTICAL_DEGREES_OFFSET) 
                        right_vert = (abs(right_tape_y - HORIZON_Y) / VERTICAL_PXL_PER_DEGREE)+ math.radians(VERTICAL_DEGREES_OFFSET) 
                                  

#-----------------------------------------------Basic Distances-------------------------------------------#
                        #Finds the distances to the left and right tapes by using the same function and switching vertical angle with left and right angles. 
                        if direction == "hatch":  # Low Camera!
                            distance_to_left_tape = (TARGET_HEIGHT - CAMERA_HEIGHT) /((math.tan(math.radians(left_vert)))) #39 
                            distance_to_right_tape = (TARGET_HEIGHT - CAMERA_HEIGHT) /((math.tan(math.radians(right_vert))))  #42
                        else:
                            distance_to_left_tape = (CAMERA_HEIGHT - TARGET_HEIGHT) /((math.tan(math.radians(left_vert))))
                            distance_to_right_tape = (CAMERA_HEIGHT - TARGET_HEIGHT) /((math.tan(math.radians(right_vert))))

#---------------------------------------Zig Zag method calculations for distance--------------------------#
                        #Our calculation to turn go forward and turn 
                        df = distance_to_right_tape
                        dn = distance_to_left_tape
                        ang = math.asin(math.sin(math.radians(angle_to_right_tape))*dn/df)
                        q=11.0/(2*math.cos(ang))
                        z=df-q
                        if headless == False:
                            cv2.putText(outimg, "Dock " + str(target_number),(cx-20,cy+10), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                            if headless == False and debug == True:
                                cv2.putText(outimg, "Distance between the turns is: " + str(round(z,2)), (360, 95), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Angle to Left Tape = " + str(round(angle_to_left_tape,2))+"'", (360, 35), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Angle to Right Tape = " + str(round(angle_to_right_tape,2))+"'", (360, 50), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Distance to right Tape = " + str(round(distance_to_right_tape,2)), (360, 80), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Distance to Left Tape = " + str(round(distance_to_left_tape,2)), (360, 65), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "left vertical angle = " + str(round(left_vert,2))+"'", (360, 115), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "right vertical angle =  " + str(round(right_vert,2)) +"'", (360, 130), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Ltape:" + str(left_tape_x) + "," + str(left_tape_y) + "; Rtape > " + str(right_tape_x) + "," + str(right_tape_y),(360, 20), font, font_size, (b, g, r), 1, cv2.LINE_AA)  
                                cv2.putText(outimg, " (" + str(cx) + "," + str(cy) + ")",(cx-40,cy+25), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, str(round(degrees_to_target-DEGREES_OFFSET,1)) +"' (" + str(round(degrees_to_target)) + " Real)",(cx-60,cy+42), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                        if degrees_to_target < -2:
                            cv2.putText(outimg, "<--", (cx+100, cy), 4, font_size*3, (255,20,255), 4, cv2.LINE_AA)
                            # cv2.putText(outimg, "RIGHT", (cx+100, cy-40), 4, font_size, (255,20,255), 1, cv2.LINE_AA)
                        elif degrees_to_target > 2:
                            # cv2.putText(outimg, "LEFT", (cx+100, cy-40), 4, font_size*5, (255,20,255), 2, cv2.LINE_AA)
                            cv2.putText(outimg, "-->", (cx+100, cy),4, font_size*3, (255,20,255), 4, cv2.LINE_AA)

#---------------------------------------------------------Hard Trig Stuff <3-----------------------------------------------#
                #-------------------------------------BASIC SIX VALUES------------------------#
                        ID = 18 #distance to stop at 
                        LA = angle_to_left_tape
                        #LA = 23.06
                        #LD = 37.74
                        LD = distance_to_left_tape
                        RA = angle_to_right_tape
                        #RA= 38.05
                        #RD=38.62
                        RD = distance_to_right_tape
                        TG = 10.0 #distance between tapes
                        DA = abs(LA-RA)
                 #--------------------------------INTERMEDIATE CALCULATIONS-------------------#
                        """ minD = min(LD,RD)
                        MD = max(LD,RD)
                        if MD == LD:
                            MA = LA
                        else:
                            MA = RA
                        cv2.putText(outimg, "JEEPERS:  " + str(minD*math.sin(math.radians(DA))/TG), (360, 180), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                       # BETA = math.degrees(math.asin(minD*math.sin(math.radians(DA))/TG))
                       # Y = TG/(2*math.cos(math.asin(minD/TG*math.sin(math.radians(DA)))))
                        #X2 = Y*math.sin(math.radians((BETA)))
                        #X = Y*math.sin(math.radians(BETA))
                        #m = math.cos(math.radians(BETA))*(ID-X)
                        #n = math.sin(math.radians(BETA))*(ID-X)
                        #DELTA = math.degrees(math.atan(m/(MD-Y-n)))
                        cv2.putText(outimg, "INTERMEDIATE VALUES", (25,125), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "minD: " + str(minD), (50, 160), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "MA: " + str(MA), (50,180), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "MD: " + str(MD), (50, 200), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "Y: " + str(round(Y,2)), (50, 260), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "X: " + str(round(X,2)), (50, 220), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "m: " + str(round(m,2)), (50, 280), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "n: " + str(round(n,2)), (50, 300), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "DELTA: " + str(round(DELTA,2)), (50, 320), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "BETA: " + str(round(BETA,2)), (50, 140), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "X2: " + str(round(X2,2)), (50, 240), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "DA: " + str(round(DA,2)), (150, 180), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                 #-----------------------OUTPUT VALUES--TURNS AND DISTANCES-------------------#
                        D4 = m/math.sin(math.radians(DELTA))
                        #THIS WILL BE OUR FIRST TURN IN DEGREES 
                        if MD == RD:
                            THETA = MA + DELTA
                        else:
                            THETA = MA - DELTA
                        if MD == LD:
                            GAMMA = 90 - BETA + DELTA
                        else:
                            GAMMA = BETA - DELTA - 90
                        D5 = ID
                        cv2.putText(outimg, "OUTPUTS", (30, 340), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "Theta: " + str(round(THETA)), (50, 360), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "D4: " + str(round(D4)), (50, 380), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "Gamma: " + str(round(GAMMA)), (50, 400), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "D5: " + str(round(D5)), (50, 420), font, font_size, (b, g, r), 1, cv2.LINE_AA)

"""#--------------------------------------------------Display art--------------------------------------------#
                        total = 0
                        for ctr in range(len(tape_keeper)):
                            total = total + degrees_to_target
                        for ctr in range(len(tape_keeper)):
                            x_val = tape_keeper[ctr][0][0]
                            y_val = tape_keeper[ctr][0][1]
                            #THE BEAUTIFUL TARGETING CIRCLES OF AWESOMENESS!! This makes the fancy fighter-jet-like display. :D
                            if headless == False:
                                if -5 < degrees_to_target < 5: 
                                    cv2.circle(outimg,(cx,cy),100,(0,255,0),thickness=2,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),75,(0,255,0),thickness=1,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),50,(0,255,0),thickness=1,lineType=8,shift=0)
                                    cv2.putText(outimg, "-", (cx-66, cy), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy-60), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy+66), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "-", (cx+60, cy), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                   
                                    if str(round(time.process_time(),1))[-1] == "0" or str(round(time.process_time(),1))[-1] == "1" or str(round(time.process_time(),1))[-1] == "4" or str(round(time.process_time(),1))[-1] == "7":
                                        cv2.circle(outimg,(cx,cy),50,(0,255,0),thickness=3,lineType=8,shift=0)
                                    elif str(round(time.process_time(),1))[-1] == "0" or str(round(time.process_time(),1))[-1] == "2" or str(round(time.process_time(),1))[-1] == "5" or str(round(time.process_time(),1))[-1] == "8":
                                        cv2.circle(outimg,(cx,cy),100,(0,255,0),thickness=3,lineType=8,shift=0)
                                    elif str(round(time.process_time(),1))[-1] == "0" or str(round(time.process_time(),1))[-1] == "3" or str(round(time.process_time(),1))[-1] == "6" or str(round(time.process_time(),1))[-1] == "9":
                                        cv2.circle(outimg,(cx,cy),75,(0,255,0),thickness=3,lineType=8,shift=0)    

                                elif -15 < degrees_to_target < 50:
                                    cv2.circle(outimg,(cx,cy),100,(0,255,255),thickness=2,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),75,(0,255,255),thickness=1,lineType=8,shift=0)
                                    cv2.putText(outimg, "-", (cx-60, cy), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy-60), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy+60), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "-", (cx+60, cy), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                        
                                else: 
                                    cv2.circle(outimg,(cx,cy),100,(0,0,255),thickness=2,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),75,(0,0,255),thickness=1,lineType=8,shift=0)
                                    cv2.putText(outimg, "-", (cx-70, cy), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy-63), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy+71), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "-", (cx+65, cy), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                
            #if headless == False: cv2.circle(outimg,(closest_dock_cx,closest_dock_cy),100,(0,255,0),thickness=1,lineType=8,shift=0)
            #if headless == False: cv2.putText(outimg, "CY IS THIS!!!:  " + str(closest_dock_cy), (360, 160), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            #cv2.putText(outimg, "Good Polygons -> " + str(len(good_polygons)) + " with center of X=" + str(cx) + ", Y=" + str(cy), (3, 40), font, font_size, (255,255,255), 1, cv2.LINE_AA)
            #else:
            #    jevois.sendSerial("-99.99")
            #code for the pig ears and snout :D
            if str(round(time.process_time()))[-1] == "0" or str(round(time.process_time()))[-1] == "1" or str(round(time.process_time()))[-1] == "4" or str(round(time.process_time()))[-1] == "7":
                if headless == False: cv2.putText(outimg, "Team250 Piggy Vision (OO)~", (400, 475), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            elif str(round(time.process_time()))[-1] == "0" or str(round(time.process_time()))[-1] == "2" or str(round(time.process_time()))[-1] == "5" or str(round(time.process_time()))[-1] == "8":
                if headless == False: cv2.putText(outimg, "Team250 Piggy Vision (OO)`", (400, 475), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            elif str(round(time.process_time()))[-1] == "0" or str(round(time.process_time()))[-1] == "3" or str(round(time.process_time()))[-1] == "6" or str(round(time.process_time()))[-1] == "9":
                if headless == False: cv2.putText(outimg, "Team250 Piggy Vision (OO)^", (400, 475), font, font_size, (b, g, r), 1, cv2.LINE_AA)
 
            # tx is center of target's x; ty is center of target's y
     
            if headless == False and debug == True: cv2.putText(outimg, "Turn -> " + str(round(closest_dock,2)) + " degrees to target", (25, 15), font, font_size, (b, g, r), 1, cv2.LINE_AA)
             
            #distance = 251.1496923*(q)/abs((240-cy))
            
           # left_vert = ((abs((left_tape_y)) / VERTICAL_PXL_PER_DEGREE)) 
           # right_vert = ((abs((right_tape_y)) / VERTICAL_PXL_PER_DEGREE))
                        
            if direction == "hatch":  # Low Camera!
                vertical_angle = ((abs((cy-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)) 
                distance = ((TARGET_HEIGHT - CAMERA_HEIGHT) /(math.tan(math.radians(vertical_angle))))# - camera_distance_from_front_bumper
#                distance = ((TARGET_HEIGHT - CAMERA_HEIGHT) /(math.tan(math.radians(vertical_angle)+math.radians(VERTICAL_DEGREES_OFFSET)))) - camera_distance_from_front_bumper
            else: # High Camera!
                vertical_angle = ((abs((cy-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)) 
                distance = ((CAMERA_HEIGHT - TARGET_HEIGHT) /(math.tan(math.radians(vertical_angle))))
                
            #distance = 15.0 * (8.5/(np.tan(vertical_angle*3.14159/180)))/(-12)
            #distance = (TARGET_HEIGHT - CAMERA_HEIGHT)/(np.tan(((cy*(.5 * VERTICAL_FOV)+vertical_angle)*3.14159/180)))
            if headless == False and debug == True: cv2.putText(outimg, "Distance From Target -> " + str(round(distance,2)), (25, 30), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            if headless == False and debug == True: cv2.putText(outimg, "Center Vertical Angle -> " + str(round(vertical_angle,2)), (25, 45), font, font_size, (b, g, r), 1, cv2.LINE_AA)
 
 #--------------------------------------Accounting for offset of camera in angles---------------------------------#
                #The following code calculates the angle to the robot's center instead of the camera.
            Ha=closest_dock
            d1=distance
            j=12.0 #distance between camera and robot center in inches
            d2=math.sqrt(math.pow(j,2)+math.pow(d1,2)-(2*d1*j*math.cos(math.radians(Ha))))
            newAngle=math.degrees(math.asin(math.sin(math.radians(Ha))* d1/d2)) 
                    
            if headless == False and debug == True: cv2.putText(outimg, "D2 = " + str(round(d2,2)) + "; newAngle: " + str(round(newAngle,2)), (25, 75
            ), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                
                # which contour, 0 is first
                #toSend = ("Dock_Target: " + str(i) +  
                #    ", x:" + str(cx) +  # center x point of target
                #    ", y:" + str(cy) +  # center y point of target
                #     ", a:" + str(degrees_to_target) +  # angle to targer
                #     ", d: Who knows!") 
                #jevois.sendSerial(toSend)
            
                #jevois.sendSerial("Angle:" + str(round(closest_dock,2)) + ",Distance:" + str(round(distance,0)))
            jevois.sendSerial("Angle:" + str(round(newAngle,2)) + ",Distance:" + str(round(distance,0)))
            cv2.putText(outimg, "A: " + str(round(newAngle,2)), (25, 110), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            cv2.putText(outimg, "D: " + str(round(distance,0)), (85, 110), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                #json_string_to_send = {"Angle" : round(closest_dock,2), "Distance" : round(distance,0)}
             
                #for ctr in range(95,-1,-1):
                #    jevois.sendSerial(str(ctr))
                #    time.sleep(1000)
                                                                                                 
                # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
            fps = self.jevois_timer.stop()
        
                #height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
            height, width, channels = outimg.shape
            if headless == False: cv2.putText(outimg, fps, (3, height - 6), font, font_size, (b, g, r), 1, cv2.LINE_AA)

        else:
            if headless == False: cv2.putText(outimg, "*** No target in view! ***", (250, 240), font, font_size, (b,g,r), 1, cv2.LINE_AA)            
            jevois.sendSerial("Angle:-99.99,Distance:-99.99")

            # Convert our BGR output image to video output format and send to host over USB. If your output image is not
            # BGR, you can use sendCvGRAY(), sendCvRGB(), or sendCvRGBA() as appropriate:
        
        #outframe.sendCv(outimg) 
        outframe.sendCvBGR(outimg,25) #COMPRESSION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #outframe.sendCvBGR(outimg)
        #outframe.sendCvGRAY(outimg)

    # ############################################### DONT YOU DARE EDIT ANYTHING IN HERE!!!  ####################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        headless = True

# COPY TO HEADLESS FROM HERE      
        debug = True
        
        start = time.time()
        
        # Constantly look and acquire color camera images 

        source0 = inimg = inframe.getCvBGR()
        outimg = inimg
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.jevois_timer.start()

        ## BEGIN GRIP CODE

        # Runs the OpenCV pipeline and sets all outputs to new values.

        self.__blur_input = source0
        (self.blur_output) = self.__blur(self.__blur_input, "Gaussian_Blur", self.__blur_radius)

        # Step CV_extractChannel0:
        self.__cv_extractchannel_src = self.blur_output
        (self.cv_extractchannel_output) = self.__cv_extractchannel(self.__cv_extractchannel_src, self.__cv_extractchannel_channel)

        # Step CV_Threshold0:
        self.__cv_threshold_src = self.cv_extractchannel_output
        (self.cv_threshold_output) = self.__cv_threshold(self.__cv_threshold_src, self.__cv_threshold_thresh, self.__cv_threshold_maxval, self.__cv_threshold_type)

        # Step Mask0:
        self.__mask_input = self.blur_output
        self.__mask_mask = self.cv_threshold_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)

        # Step Normalize0:
        self.__normalize_input = self.mask_output
        (self.normalize_output) = self.__normalize(self.__normalize_input, self.__normalize_type, self.__normalize_alpha, self.__normalize_beta)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.normalize_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_erode0:
        self.__cv_erode_src = self.hsv_threshold_output
        (self.cv_erode_output) = self.__cv_erode(self.__cv_erode_src, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)

        # Step CV_dilate0:
        self.__cv_dilate_src = self.cv_erode_output
        (self.cv_dilate_output) = self.__cv_dilate(self.__cv_dilate_src, self.__cv_dilate_kernel, self.__cv_dilate_anchor, self.__cv_dilate_iterations, self.__cv_dilate_bordertype, self.__cv_dilate_bordervalue)

        # Step Find_Contours0:
        self.__find_contours_input = self.cv_dilate_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)

        ### End Grip Code
       
        ### Start Non-Grip Common/Default Code

        def sort_contours(cnts, method = "left-to-right"):
            
            jevois.LINFO("In sort contours!" + str(len(cnts)))
                            
            reverse = False
            i=0
        
            if method == "right-to-left" or method == "bottom-to-top":
                reverse = True
            
            if method == "top-to-bottom" or method == "bottom-to-top":
                i=i
            
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
        
            return (cnts, boundingBoxes)    

        def draw_contour(image, c, i):
            M = cv2.moments(c)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
        
            cv2.putText(image, "#{}".format(i+1),(cX-20,cY),font, font_size, (255,255,255), 2)
        
            return image
             
        ### Start Additional Non-Grip Custom Code
          
# Constants and Variables---The following is all of our constants and variables we use in our formulas later in the code. :D
        b = 61
        g = 260
        r = 148
        direction = "hatch"
        font = 0
        font_size = .4      
        degrees_to_target = 0
        w, h = 2, 2;
        contour_keeper = [[0 for x in range(w)] for y in range(h)] 
        tape_keeper = [] 
        cx = 0
        cy = 0
        closest_dock = -99.99
        VERTICAL_FOV = 52
        VERTICAL_PXL_PER_DEGREE = 6.75
        HORIZON_Y = 450
        camera_distance_from_front_bumper = 5
        #SETTING VALUES FOR HATCH VS CARGO :D
        if direction == "hatch":
            CAMERA_X_OFFSET = 12.5
            CAMERA_HEIGHT = 10.00
            TARGET_HEIGHT = 31.5
            CALIBRATION_FACTOR = 0
            DEGREES_OFFSET = -17
            VERTICAL_DEGREES_OFFSET = 28.9
        else:
            CAMERA_X_OFFSET = 0  
            CAMERA_HEIGHT = 46.0
            TARGET_HEIGHT = 31.5
            CALIBRATION_FACTOR = -30
            DEGREES_OFFSET = 7
                
        # Draws all contours on original image in red
        cv2.drawContours(outimg, self.filter_contours_output, -1, (0, 0, 250), 1)
        
        # Draw screen grid #This is the code that sets up a grip pattern on the display screen. Green ticks will be placed at every 10 pixels while red ticks will be placed every 100.
        if headless == False and debug == True:
            # X axis grid
            for pixel in range(100, 640, 100):
                cv2.putText(outimg, "||", (pixel, 455), font, font_size, (0, 0, 255), 1, cv2.LINE_AA)
            for pixel in range(30, 640, 10):
                cv2.putText(outimg, "|", (pixel, 455), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            for pixel in range(30, 640, 10):
                cv2.putText(outimg, str(round((pixel%100)*.1)), (pixel, 445), font, font_size, (b, g, r), 1, cv2.LINE_AA)

            # Y axis grid        
            for pixel in range(100, 440, 100):
                cv2.putText(outimg, "=", (1,pixel), font, font_size, (0, 0, 255), 1, cv2.LINE_AA)
            for pixel in range(0, 440, 10):
                cv2.putText(outimg, "-", (1,pixel), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            for pixel in range(0, 440, 10):
                cv2.putText(outimg, str(round((pixel%100)*.1)), (15,pixel), font, font_size, (b, g, r), 1, cv2.LINE_AA)
       
        # Gets number of contours
        NumOfContours = len(self.filter_contours_output)
        #if headless == False: cv2.putText(outimg, "Contours Found: " + str(NumOfContours), (3, 450), font, font_size, (b, g, r), 1, cv2.LINE_AA)
        
        # Sorts contours from left to right
        #newContours = sortByArea(self.filter_contours_output)  # Sorts contours by area
        newContours = sorted(self.filter_contours_output, key=lambda ctr: cv2.boundingRect(ctr)[0])
                               
        #Organize the contours we found
        jevois.LINFO("About to run through contours: " + str(NumOfContours) + " to see if the are tapes")
        if NumOfContours != 0:                   # If there are things to analyze in the image...
            for i in range (NumOfContours):      # Loop through everything found...
                cnt = newContours[i]
                
                rect = cv2.minAreaRect(cnt)
                
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                x,y,theta = cv2.minAreaRect(cnt)

                jevois.LINFO("Contour: " + str(i))
                jevois.LINFO("=================")
                jevois.LINFO("x[0] or CX = " + str(x[0]))
                jevois.LINFO("x[1] or CY = " + str(x[1]))
                jevois.LINFO("y[0] or W = " + str(y[0]))
                jevois.LINFO("y[1] or H = " + str(y[1]))
                jevois.LINFO("angle = " + str(theta))
                                                
                cv2.drawContours(outimg,[box],0,(0,255,0),2)
                if headless == False and debug == True: cv2.putText(outimg, "C" + str(i) + " at " + str(round(theta)) + "'", (round(x[0]-30),round(x[1]+80)), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                #cv2.Point2f rect_points[4]
                
                #x,y,w,h = cv2.boundingRect(cnt) # Get the stats of the contour including width and height
                #cv2.rectangle(outimg,(x,y),(x+w,y+h),(0,255,0),2)  #this draws the bounding rectangle in our image :)
                #contour_keeper[i] = x 
                cnt_area = cv2.contourArea(cnt)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                p = cv2.approxPolyDP(hull, -1, 1)
        
                # Determine if something is a piece of tape correctly angled or not
                #Theta is the angle of the rectangle returned by the minAreaRect method imported above :D
                if (theta > -30 and theta < -4) and theta != 0.0:
                    tape_keeper.insert(i,[x,y,"L"])
                elif (theta > -85 and theta < -60) and theta != 0.0:
                    tape_keeper.insert(i,[x,y,"R"])
   
            # Now we only have angled good tapes, so we loop through and identify docks
            if len(tape_keeper) != 0:
                 jevois.LINFO("Looping through " + str(len(tape_keeper)) + " found tapes and determining docks.")
                 if headless == False and debug == True: cv2.putText(outimg, "Tracking " + str(len(tape_keeper)) + " tapes!",(25,60), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                 left_side = False      # flag indicating just saw a left (right leaning) piece of tape
                 right_side = False     # flag indicating just saw a right (left leaning) piece of tape
                 target_found = False   # flag indicating we found a left and right pair - a port!
                 target_number = 0      # number of pairs found

                 t = len(tape_keeper)   # t gets the number of pieces of tape.                 
                 
                 # Loop through the number of found pairs of tape
                 for ctr in range(t):
                     jevois.LINFO("Looking at a " + tape_keeper[ctr][2])

                     # If it's a left side, indicate it's point right...
                     if str(tape_keeper[ctr][2]) == "R":
                        left_side = True
                     # If it's a right side, indicate it's pointing left...
                     elif str(tape_keeper[ctr][2]) == "L" and left_side == True:
                        target_found = True
                        target_number = target_number + 1
                        left_side = False
                        right_side = False
                        jevois.LINFO("Found a full target!!!!")
                        
                        # Save the center X and center Y for this particular matching pair
                        cx = round(tape_keeper[ctr-1][0][0] + (.5 * (tape_keeper[ctr][0][0]-tape_keeper[ctr-1][0][0]))) + round(CAMERA_X_OFFSET)
                        #cy = round((tape_keeper[ctr][0][1]+tape_keeper[ctr-1][0][1])/2)
                        
                        # THIS IS WHERE WE SET THE X AND Y VALUES OF THE TARGET'S TAPES :D
                        left_tape_x = round(tape_keeper[ctr-1][0][0]+tape_keeper[ctr-1][1][1]) #The value returned as the x coordinate is the upper left hand corner, so we must add the width of the min area rectangle.
                        left_tape_y = round(tape_keeper[ctr-1][0][1])+round(tape_keeper[ctr-1][1][1])/2
                        right_tape_x = round(tape_keeper[ctr][0][0])-33
                        right_tape_y = round(tape_keeper[ctr][0][1])+round(tape_keeper[ctr-1][1][1])/2
                        cy = round((right_tape_y+left_tape_y)/2) #This takes the average of the y values to find the center y value of the target :D
                        
            #--------------------------------------------HORIZONTAL ANGLES--------------------------------------------#
                        #Calculates the horizontal degrees to target using the difference between center x and the center of the image.
                        degrees_to_target = (cx - 320)/9 - (CAMERA_X_OFFSET/ 9) + DEGREES_OFFSET
                        #The following finds the horizontal angles to each tapes by subtracting the x values of each and finding the distance in pixels to the center of the image. 
                        #The 9 is the number of pixels per degree in the FOV. 
                        angle_to_left_tape = (left_tape_x - 320)/9 - (CAMERA_X_OFFSET/ 9) + DEGREES_OFFSET
                        angle_to_right_tape = (right_tape_x - 320)/9 - (CAMERA_X_OFFSET/ 9) + DEGREES_OFFSET
                        
                        if abs(degrees_to_target) < abs(closest_dock):
                            closest_dock = degrees_to_target
                            closest_dock_cx = cx
                            closest_dock_cy = cy
                           
#-------------------Same thing as above but consolidated into one loop for calculating and displaying distance and vertical angle-----------#
            if closest_dock != -99.99:

           #---------------------------------------------VERTICAL ANGLES------------------------------------------------#
                        if headless == False and debug == True: cv2.putText(outimg, "cy = " + str(cy) + " horizon_y " + str(HORIZON_Y),(25,90), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        if headless == False and debug == True: cv2.line(outimg,(0,HORIZON_Y),(640,HORIZON_Y),(0,0,255),thickness=1,lineType=8,shift=0)
                        
                        # Left Tape angle
                        #The following finds the vertical angle of the target to the robot by taking the y coordinate of the tape and subtracting 405 pixels.
                        #the horizon_y = 405 is simply taking into account the camera's tilt (it's calibrated not an exact value) :D
                        #vertical_angle = ((abs((cy-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)/ 2) 

                        # Right Tape--This does the same thing but for right tapes the t-1 is to make sure you're not at the end
                        #if ctr != t - 1: vertical_angle = ((abs((tape_keeper[ctr+1][0][1]-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)/ 2) 
                        
                        #These two lines find the vertical angles of the right and left tapes of the target the same way as finding the vertical angle for turning. :D
                        left_vert = (abs(left_tape_y - HORIZON_Y) / VERTICAL_PXL_PER_DEGREE) + math.radians(VERTICAL_DEGREES_OFFSET) 
                        right_vert = (abs(right_tape_y - HORIZON_Y) / VERTICAL_PXL_PER_DEGREE)+ math.radians(VERTICAL_DEGREES_OFFSET) 
                                  

#-----------------------------------------------Basic Distances-------------------------------------------#
                        #Finds the distances to the left and right tapes by using the same function and switching vertical angle with left and right angles. 
                        if direction == "hatch":  # Low Camera!
                            distance_to_left_tape = (TARGET_HEIGHT - CAMERA_HEIGHT) /((math.tan(math.radians(left_vert)))) #39 
                            distance_to_right_tape = (TARGET_HEIGHT - CAMERA_HEIGHT) /((math.tan(math.radians(right_vert))))  #42
                        else:
                            distance_to_left_tape = (CAMERA_HEIGHT - TARGET_HEIGHT) /((math.tan(math.radians(left_vert))))
                            distance_to_right_tape = (CAMERA_HEIGHT - TARGET_HEIGHT) /((math.tan(math.radians(right_vert))))

#---------------------------------------Zig Zag method calculations for distance--------------------------#
                        #Our calculation to turn go forward and turn 
                        df = distance_to_right_tape
                        dn = distance_to_left_tape
                        ang = math.asin(math.sin(math.radians(angle_to_right_tape))*dn/df)
                        q=11.0/(2*math.cos(ang))
                        z=df-q
                        if headless == False:
                            cv2.putText(outimg, "Dock " + str(target_number),(cx-20,round(tape_keeper[ctr][0][1])), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                            if headless == False and debug == True:
                                cv2.putText(outimg, "Distance between the turns is: " + str(round(z,2)), (360, 95), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Angle to Left Tape = " + str(round(angle_to_left_tape,2))+"'", (360, 35), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Angle to Right Tape = " + str(round(angle_to_right_tape,2))+"'", (360, 50), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Distance to right Tape = " + str(round(distance_to_right_tape,2)), (360, 80), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Distance to Left Tape = " + str(round(distance_to_left_tape,2)), (360, 65), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "left vertical angle = " + str(round(left_vert,2))+"'", (360, 115), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "right vertical angle =  " + str(round(right_vert,2)) +"'", (360, 130), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, "Ltape:" + str(left_tape_x) + "," + str(left_tape_y) + "; Rtape > " + str(right_tape_x) + "," + str(right_tape_y),(360, 20), font, font_size, (b, g, r), 1, cv2.LINE_AA)  
                                cv2.putText(outimg, " (" + str(cx) + "," + str(cy) + ")",(cx-30,round(tape_keeper[ctr][0][1])+20), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                                cv2.putText(outimg, str(round(degrees_to_target-DEGREES_OFFSET,1)) +"' (" + str(round(degrees_to_target)) + " Real)",(cx-40,round(tape_keeper[ctr][0][1])+40), font, font_size, (b, g, r), 1, cv2.LINE_AA)

                        if degrees_to_target > 2:
                            cv2.putText(outimg, "<--", (cx-50, cy+130), 4, font_size*5, (255,20,255), 4, cv2.LINE_AA)
                            # cv2.putText(outimg, "RIGHT", (cx+100, cy-40), 4, font_size, (255,20,255), 1, cv2.LINE_AA)
                        elif degrees_to_target < -2:
                            # cv2.putText(outimg, "LEFT", (cx+100, cy-40), 4, font_size*5, (255,20,255), 2, cv2.LINE_AA)
                            cv2.putText(outimg, "-->", (cx-50, cy+130),4, font_size*5, (255,20,255), 4, cv2.LINE_AA)

#---------------------------------------------------------Hard Trig Stuff <3-----------------------------------------------#
                #-------------------------------------BASIC SIX VALUES------------------------#
                        ID = 18 #distance to stop at 
                        LA = angle_to_left_tape
                        #LA = 23.06
                        #LD = 37.74
                        LD = distance_to_left_tape
                        RA = angle_to_right_tape
                        #RA= 38.05
                        #RD=38.62
                        RD = distance_to_right_tape
                        TG = 10.0 #distance between tapes
                        DA = abs(LA-RA)
                 #--------------------------------INTERMEDIATE CALCULATIONS-------------------#
                        minD = min(LD,RD)
                        MD = max(LD,RD)
                        if MD == LD:
                            MA = LA
                        else:
                            MA = RA
                        cv2.putText(outimg, "JEEPERS:  " + str(minD*math.sin(math.radians(DA))/TG), (360, 180), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        BETA = math.degrees(math.asin(minD*math.sin(math.radians(DA))/TG))
                        Y = TG/(2*math.cos(math.asin(minD/TG*math.sin(math.radians(DA)))))
                        X2 = Y*math.sin(math.radians((BETA)))
                        X = Y*math.sin(math.radians(BETA))
                        m = math.cos(math.radians(BETA))*(ID-X)
                        n = math.sin(math.radians(BETA))*(ID-X)
                        DELTA = math.degrees(math.atan(m/(MD-Y-n)))
                        cv2.putText(outimg, "INTERMEDIATE VALUES", (25,125), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "minD: " + str(minD), (50, 160), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "MA: " + str(MA), (50,180), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "MD: " + str(MD), (50, 200), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "Y: " + str(round(Y,2)), (50, 260), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "X: " + str(round(X,2)), (50, 220), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "m: " + str(round(m,2)), (50, 280), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "n: " + str(round(n,2)), (50, 300), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "DELTA: " + str(round(DELTA,2)), (50, 320), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "BETA: " + str(round(BETA,2)), (50, 140), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "X2: " + str(round(X2,2)), (50, 240), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "DA: " + str(round(DA,2)), (150, 180), font, font_size, (b, g, r), 1, cv2.LINE_AA)


                 #-----------------------OUTPUT VALUES--TURNS AND DISTANCES-------------------#
                        D4 = m/math.sin(math.radians(DELTA))
                        #THIS WILL BE OUR FIRST TURN IN DEGREES 
                        if MD == RD:
                            THETA = MA + DELTA
                        else:
                            THETA = MA - DELTA
                        if MD == LD:
                            GAMMA = 90 - BETA + DELTA
                        else:
                            GAMMA = BETA - DELTA - 90
                        D5 = ID
                        cv2.putText(outimg, "OUTPUTS", (30, 340), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "Theta: " + str(round(THETA)), (50, 360), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "D4: " + str(round(D4)), (50, 380), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "Gamma: " + str(round(GAMMA)), (50, 400), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                        cv2.putText(outimg, "D5: " + str(round(D5)), (50, 420), font, font_size, (b, g, r), 1, cv2.LINE_AA)
#--------------------------------------------------Display art--------------------------------------------#
                        total = 0
                        for ctr in range(len(tape_keeper)):
                            total = total + degrees_to_target
                        for ctr in range(len(tape_keeper)):
                            x_val = tape_keeper[ctr][0][0]
                            y_val = tape_keeper[ctr][0][1]
                            #THE BEAUTIFUL TARGETING CIRCLES OF AWESOMENESS!! This makes the fancy fighter-jet-like display. :D
                            if headless == False:
                                if -5 < degrees_to_target < 5: 
                                    cv2.circle(outimg,(cx,cy),100,(0,255,0),thickness=2,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),70,(0,255,0),thickness=1,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),40,(0,255,0),thickness=1,lineType=8,shift=0)
                                    cv2.putText(outimg, "-", (cx-69, cy), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy-60), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy+67), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "-", (cx+58, cy), font, font_size, (0, 255, 0), 1, cv2.LINE_AA)
                   
                                    if str(round(time.process_time(),1))[-1] == "0" or str(round(time.process_time(),1))[-1] == "1" or str(round(time.process_time(),1))[-1] == "4" or str(round(time.process_time(),1))[-1] == "7":
                                        cv2.circle(outimg,(cx,cy),40,(0,255,0),thickness=3,lineType=8,shift=0)
                                    elif str(round(time.process_time(),1))[-1] == "0" or str(round(time.process_time(),1))[-1] == "2" or str(round(time.process_time(),1))[-1] == "5" or str(round(time.process_time(),1))[-1] == "8":
                                        cv2.circle(outimg,(cx,cy),100,(0,255,0),thickness=3,lineType=8,shift=0)
                                    elif str(round(time.process_time(),1))[-1] == "0" or str(round(time.process_time(),1))[-1] == "3" or str(round(time.process_time(),1))[-1] == "6" or str(round(time.process_time(),1))[-1] == "9":
                                        cv2.circle(outimg,(cx,cy),70,(0,255,0),thickness=3,lineType=8,shift=0)    
                                elif -15<degrees_to_target<50:
                                    cv2.circle(outimg,(cx,cy),100,(0,255,255),thickness=2,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),70,(0,255,255),thickness=1,lineType=8,shift=0)
                                    cv2.putText(outimg, "-", (cx-69, cy), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy-60), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy+67), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "-", (cx+58, cy), font, font_size, (0,255,255), 1, cv2.LINE_AA)
                                else: 
                                    cv2.circle(outimg,(cx,cy),100,(0,0,255),thickness=2,lineType=8,shift=0)
                                    cv2.circle(outimg,(cx,cy),70,(0,0,255),thickness=1,lineType=8,shift=0)
                                    cv2.putText(outimg, "-", (cx-69, cy), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy-60), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "|", (cx, cy+67), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                    cv2.putText(outimg, "-", (cx+58, cy), font, font_size, (0,0,255), 1, cv2.LINE_AA)
                                
            #if headless == False: cv2.circle(outimg,(closest_dock_cx,closest_dock_cy),100,(0,255,0),thickness=1,lineType=8,shift=0)
            #if headless == False: cv2.putText(outimg, "CY IS THIS!!!:  " + str(closest_dock_cy), (360, 160), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            #cv2.putText(outimg, "Good Polygons -> " + str(len(good_polygons)) + " with center of X=" + str(cx) + ", Y=" + str(cy), (3, 40), font, font_size, (255,255,255), 1, cv2.LINE_AA)
            #else:
            #    jevois.sendSerial("-99.99")
            #code for the pig ears and snout :D
            if str(round(time.process_time()))[-1] == "0" or str(round(time.process_time()))[-1] == "1" or str(round(time.process_time()))[-1] == "4" or str(round(time.process_time()))[-1] == "7":
                if headless == False: cv2.putText(outimg, "Team250 Piggy Vision (OO)~", (450, 470), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            elif str(round(time.process_time()))[-1] == "0" or str(round(time.process_time()))[-1] == "2" or str(round(time.process_time()))[-1] == "5" or str(round(time.process_time()))[-1] == "8":
                if headless == False: cv2.putText(outimg, "Team250 Piggy Vision (OO)`", (450, 470), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            elif str(round(time.process_time()))[-1] == "0" or str(round(time.process_time()))[-1] == "3" or str(round(time.process_time()))[-1] == "6" or str(round(time.process_time()))[-1] == "9":
                if headless == False: cv2.putText(outimg, "Team250 Piggy Vision (OO)^", (450, 470), font, font_size, (b, g, r), 1, cv2.LINE_AA)
 
            # tx is center of target's x; ty is center of target's y
     
            if headless == False and debug == True: cv2.putText(outimg, "Turn -> " + str(round(closest_dock,2)) + " degrees to target", (25, 15), font, font_size, (b, g, r), 1, cv2.LINE_AA)
             
            #distance = 251.1496923*(q)/abs((240-cy))
            
           # left_vert = ((abs((left_tape_y)) / VERTICAL_PXL_PER_DEGREE)) 
           # right_vert = ((abs((right_tape_y)) / VERTICAL_PXL_PER_DEGREE))
                        
            if direction == "hatch":  # Low Camera!
                vertical_angle = ((abs((cy-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)) 
                distance = ((TARGET_HEIGHT - CAMERA_HEIGHT) /(math.tan(math.radians(vertical_angle))))# - camera_distance_from_front_bumper
#                distance = ((TARGET_HEIGHT - CAMERA_HEIGHT) /(math.tan(math.radians(vertical_angle)+math.radians(VERTICAL_DEGREES_OFFSET)))) - camera_distance_from_front_bumper
            else: # High Camera!
                vertical_angle = ((abs((cy-HORIZON_Y)) / VERTICAL_PXL_PER_DEGREE)) 
                distance = ((CAMERA_HEIGHT - TARGET_HEIGHT) /(math.tan(math.radians(vertical_angle))))
                
            #distance = 15.0 * (8.5/(np.tan(vertical_angle*3.14159/180)))/(-12)
            #distance = (TARGET_HEIGHT - CAMERA_HEIGHT)/(np.tan(((cy*(.5 * VERTICAL_FOV)+vertical_angle)*3.14159/180)))
            if headless == False and debug == True: cv2.putText(outimg, "Distance From Target -> " + str(round(distance,2)), (25, 30), font, font_size, (b, g, r), 1, cv2.LINE_AA)
            if headless == False and debug == True: cv2.putText(outimg, "Center Vertical Angle -> " + str(round(vertical_angle,2)), (25, 45), font, font_size, (b, g, r), 1, cv2.LINE_AA)
 
 #--------------------------------------Accounting for offset of camera in angles---------------------------------#
                #The following code calculates the angle to the robot's center instead of the camera.
            Ha=closest_dock
            d1=distance
            j=12.0 #distance between camera and robot center in inches
            d2=math.sqrt(math.pow(j,2)+math.pow(d1,2)-(2*d1*j*math.cos(math.radians(Ha))))
            newAngle=math.degrees(math.asin(math.sin(math.radians(Ha))* d1/d2)) 
                    
            if headless == False and debug == True: cv2.putText(outimg, "D2 = " + str(round(d2,2)) + "; newAngle: " + str(round(newAngle,2)), (25, 75
            ), font, font_size, (b, g, r), 1, cv2.LINE_AA)
                
                # which contour, 0 is first
                #toSend = ("Dock_Target: " + str(i) +  
                #    ", x:" + str(cx) +  # center x point of target
                #    ", y:" + str(cy) +  # center y point of target
                #     ", a:" + str(degrees_to_target) +  # angle to targer
                #     ", d: Who knows!") 
                #jevois.sendSerial(toSend)
            
                #jevois.sendSerial("Angle:" + str(round(closest_dock,2)) + ",Distance:" + str(round(distance,0)))
            jevois.sendSerial("Angle:" + str(round(newAngle,2)) + ",Distance:" + str(round(distance,0)))
                #json_string_to_send = {"Angle" : round(closest_dock,2), "Distance" : round(distance,0)}
             
                #for ctr in range(95,-1,-1):
                #    jevois.sendSerial(str(ctr))
                #    time.sleep(1000)
                                                                                                 
                # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
            fps = self.jevois_timer.stop()
        
                #height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
            height, width, channels = outimg.shape
            if headless == False: cv2.putText(outimg, fps, (3, height - 6), font, font_size, (b, g, r), 1, cv2.LINE_AA)

        else:
            if headless == False: cv2.putText(outimg, "*** No target in view! ***", (250, 240), font, font_size, (b,g,r), 1, cv2.LINE_AA)            
            jevois.sendSerial("Angle:-99.99,Distance:-99.99")

            # Convert our BGR output image to video output format and send to host over USB. If your output image is not
            # BGR, you can use sendCvGRAY(), sendCvRGB(), or sendCvRGBA() as appropriate:
        
        outframe.sendCvBGR(outimg)
            #outframe.sendCvGRAY(outimg)
    # ###################################################################################################
    ## Parse a serial command forwarded to us by the JeVois Engine, return a string
    def parseSerial(self, str):
        jevois.LINFO("parseserial received command [{}]".format(str))
        if str == "hello":
            return self.hello()
        return "ERR Unsupported command"
    
    # ###################################################################################################
    ## Return a string that describes the custom commands we support, for the JeVois help message
    def supportedCommands(self):
        # use \n seperator if your module supports several commands
        return "hello - print hello using python"

    @staticmethod
    def __blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        ksize = int(2 * round(radius) + 1)
        return cv2.blur(src, (ksize, ksize))
        #return cv2.medianBlur(src, (ksize, ksize)) # Perform a Median Blur
        #return cv2.GaussianBlur(src,(ksize, ksize),0) # Perform a Gaussian Blur
                        
    @staticmethod
    def __cv_extractchannel(src, channel):
        """Extracts given channel from an image.
        Args:
            src: A numpy.ndarray.
            channel: Zero indexed channel number to extract.
        Returns:
             The result as a numpy.ndarray.
        """
        return cv2.extractChannel(src, (int) (channel + 0.5))

    @staticmethod
    def __cv_threshold(src, thresh, max_val, type):
        """Apply a fixed-level threshold to each array element in an image
        Args:
            src: A numpy.ndarray.
            thresh: Threshold value.
            max_val: Maximum value for THRES_BINARY and THRES_BINARY_INV.
            type: Opencv enum.
        Returns:
            A black and white numpy.ndarray.
        """
        return cv2.threshold(src, thresh, max_val, type)[1]

    @staticmethod
    def __mask(input, mask):
        """Filter out an area of an image using a binary mask.
        Args:
            input: A three channel numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            A three channel numpy.ndarray.
        """
        return cv2.bitwise_and(input, input, mask=mask)

    @staticmethod
    def __normalize(input, type, a, b):
        """Normalizes or remaps the values of pixels in an image.
        Args:
            input: A numpy.ndarray.
            type: Opencv enum.
            a: The minimum value.
            b: The maximum value.
        Returns:
            A numpy.ndarray of the same type as the input.
        """
        return cv2.normalize(input, None, a, b, type)

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_erode(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of lower value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for erosion. A numpy.ndarray.
           iterations: the number of times to erode.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after erosion.
        """
        return cv2.erode(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __cv_dilate(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of higher value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for dilation. A numpy.ndarray.
           iterations: the number of times to dilate.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after dilation.
        """
        return cv2.dilate(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        #im2, contours, hierarchy = cv2.findContours(input, mode=mode, method=method)
        contours,hierachy=cv2.findContours(input,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
        return output
        
#BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')
#YEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET YOU HAVE COMPLETED THE CODE YEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET
