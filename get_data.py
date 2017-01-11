import numpy as np
import cv2
import pickle
import time


index=0
" Load video "
vid=cv2.VideoCapture('Lane_change_%s.mp4' %(index))

file=open('car_init.p','r')
old_lines=pickle.load(file)
old_box_origin=pickle.load(file)
old_box_width=pickle.load(file)
x1_old=0
x2_old=0
y1_old=0
y2_old=0
centers=[]
distances=[]
thetas=[]

while (vid.isOpened()):
    t1=time.time()
    ret,frame=vid.read()

    " Check if an image is read "
    if ret:
        " Convert image to grayscale then perform histogram equalization"
        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = cv2.equalizeHist(gray)
        # edges=cv2.Canny(contrast,200,400)

        " Run Haar cascade classifier "
        car_detector = cv2.CascadeClassifier("cars.xml")
        boxes = car_detector.detectMultiScale(contrast, 1.05, 5)
        box_ind = np.argmax(boxes[:, 3]) # Only keep the biggest box which corresponds to the closest car
        box_origin = boxes[box_ind, 0:2]
        box_width= boxes[box_ind,3]

        if box_width < 200 :  # If the box is too small, the car is not relevant for the detector, keep box from previous frame
            lines = old_lines
            box_origin=old_box_origin
            box_width=old_box_width

        " Set ROI for lane detection"
        roi = contrast[box_origin[1]:box_origin[1] + box_width, box_origin[0]:box_origin[0] + box_width]
        car_center = [box_origin[0] + int(float(box_width) / 2), box_origin[1] + int(float(box_width) / 2)]

        render=image

        # edge=cv2.Canny(contrast,200,400)
        # newOriginX = 0
        # newOriginY = 0
        #
        #
        # """
        # Checking which side (left vs.right) of the image object is on and setting ROI origin appropriately
        # """

        # if car[0] > width/2 :
        #     roi = edge[car[0]-150:car[0]-150+car[2]+50,car[1]:car[1]+car[3]]
        #     newOriginX = car_center[0]-150
        #     newOriginY = car_center[1]
        # else :
        #     roi = edge[car_center[0]:car_center[0]+car[2] + 50, car_center[1]:car_center[1]+int(float(car[3]) / 2)]
        #     newOriginX = car_center[0]
        #     newOriginY = car_center[1]

        " Draw box around car "
        cv2.rectangle(render, (box_origin[0], box_origin[1]), (box_origin[0] + box_width, box_origin[1] + box_width), (255, 0, 0), 2)

        centers.append(car_center)
        cv2.circle(render, (car_center[0], car_center[1]), 5, (0, 0, 255), -1)




        box_edges = cv2.Canny(roi, 350, 450)
        lines = cv2.HoughLines(box_edges, 1, np.pi/180, 35)

        rayon = np.array([])
        angle = np.array([])
        for rho, theta in lines[0]:
            if  0.3<theta < np.pi / 4:
                angle = np.append(angle, theta)
                rayon = np.append(rayon, rho)

        x1 = x1_old
        x2 = x2_old
        y1 = y1_old
        y2 = y2_old

        if len(angle)>0:
            theta = np.min(angle)
            rho = rayon[np.argmin(angle)]



            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # print(theta, rho)
            # print(x0)

            if x0 > 50 :
                x1 = int(x0+ 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                x1_old = x1
                y1_old = y1
                x2_old = x2
                y2_old = y2

        cv2.line(render, (x1 + box_origin[0], y1 + box_origin[1]), (x2 + box_origin[0], y2 + box_origin[1]),(0, 255, 0), 2)
        " Keep current box for next frame"
        old_box_origin = box_origin
        old_box_width = box_width

        " Keep current line for next frame"
        old_lines = lines

        distance = rho + np.linalg.norm(box_origin)

        centers.append(car_center)
        distances.append(distance)
        thetas.append(theta)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    t2=time.time()
    t3=t2-t1
    if t3 > 2:
        break
vid.release()


"Uncomment"

file_1=open('pos_%s.p' %(index),'wb')
pickle.dump(centers,file_1)
pickle.dump(distances,file_1)
pickle.dump(thetas,file_1)
file_1.close()
