import numpy as np
import cv2
import detection



image=cv2.imread("/home/parinya/spring_17/car/python_port/positive/car2.png")
roi, image,box_origin,box_width=detection.get_cars(image)
image=detection.draw_box(image,box_origin,box_width)

image=detection.lanes(roi,image,car)



cv2.imshow('image',image)

cv2.waitKey(0)
cv2.destroyAllWindows()

