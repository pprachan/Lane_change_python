import numpy as np
import cv2
import pickle


image = cv2.imread("/home/parinya/spring_17/Car/Lane_change_python/positive/car2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contrast = cv2.equalizeHist(gray)

" Run Haar cascade classifier "
car_detector = cv2.CascadeClassifier("/home/parinya/spring_17/Car/Lane_change_python/cars.xml")
boxes = car_detector.detectMultiScale(contrast, 1.05, 5)
box_ind = np.argmax(boxes[:, 3])
box_origin = boxes[box_ind, 0:2]
box_width = boxes[box_ind, 3]

render = image

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
cv2.rectangle(render, (box_origin[0], box_origin[1]), (box_origin[0] + box_width, box_origin[1] + box_width),
              (255, 0, 0), 2)
car_center = [box_origin[0] + int(float(box_width) / 2), box_origin[1] + int(float(box_width) / 2)]
cv2.circle(render, (car_center[0], car_center[1]), 5, (0, 0, 255), -1)

" Set ROI for lane detection"
roi = contrast[box_origin[1]:box_origin[1] + box_width, box_origin[0]:box_origin[0] + box_width]


edges = cv2.Canny(roi, 350, 450)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 35)

rayon = np.array([])
angle = np.array([])

for rho, theta in lines[0]:
    if theta < np.pi / 4:
        angle = np.append(angle, theta)
        rayon = np.append(rayon, rho)

theta = np.min(angle)
rho = rayon[np.argmin(angle)]

print(theta, rho)

a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
x1 = int(x0 + 1000 * (-b))
y1 = int(y0 + 1000 * (a))
x2 = int(x0 - 1000 * (-b))
y2 = int(y0 - 1000 * (a))

cv2.line(image, (x1 + box_origin[0], y1 + box_origin[1]), (x2 + box_origin[0], y2 + box_origin[1]), (0, 255, 0), 2)

cv2.imshow('image',image)

cv2.waitKey(0)
cv2.destroyAllWindows()

file=open('car_init.p','wb')
pickle.dump(lines,file)
pickle.dump(box_origin,file)
pickle.dump(box_width,file)
file.close()