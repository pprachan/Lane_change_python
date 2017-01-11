import numpy as np
import pickle
import detection

# index=raw_input() # use index=input() for python 3

index=2

file_1=open('/home/parinya/spring_17/car/lane_change_python/positive/pos_%s.p' %(index),'rb')
centers=pickle.load(file_1)
distances=pickle.load(file_1)
thetas=pickle.load(file_1)
file_1.close()


speed=detection.derivative(distances,1/25)


car_slope=[]
old_center=[0,0]
for center in centers:
    vector_c=np.subtract(center,old_center)*25
    old_center=center
    car_slope.append(vector_c)


angle=[]
betas=[]
for vector in car_slope:
    beta = np.arctan2(vector[1],vector[0])
    betas.append(beta)
    for theta in thetas:
        angle.append(abs(theta-beta))


angle_velocity=detection.derivative(angle,1/25)

file_2=open('/home/parinya/spring_17/car/lane_change_python/features/feat_pos_%s.p' %(index),'wb')
pickle.dump(distances,file_2)
pickle.dump(speed,file_2)
pickle.dump(angle,file_2)
pickle.dump(angle_velocity,file_2)
file_2.close()


