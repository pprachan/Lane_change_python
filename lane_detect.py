import numpy as np
import cv2
import detection
" Load video "
vid=cv2.VideoCapture('/home/parinya/spring_17/car/python_port/positive/Lane_change_2.mp4')

while(vid.isOpened()):
    "Capture frame by frame"
    ret,frame=vid.read()
    if ret :
        roi, image, car = detection.cars(frame)
        image = detection.lanes(roi, image, car)
        cv2.imshow('image',roi)
        cv2.waitKey(25)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# cv2.waitKey(0)
vid.release()
cv2.destroyAllWindows()

    # img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    # cars=car_detect(image)
    # cv2.imshow('frame',cars)

    # cap = cv2.VideoCapture("/home/parinya/spring_17/car/python_port/positive/Lane_change_1.mp4")

    # while(cap.isOpened()):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     #print frame.shape

    #     if(ret): #if cam read is successfull
    #         # Our operations on the frame come here
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    	out=car_detect(gray)

    #         # Display the resulting frame
    #         cv2.imshow('frame',out)
    #     else:
    #         break

    #     # this should be called always, frame or not.
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

