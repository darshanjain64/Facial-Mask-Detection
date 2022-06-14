import cv2
import os

import numpy as np

haar_data=cv2.CascadeClassifier(r"D:\DAA Project\data.xml")
data=[]
capture=cv2.VideoCapture(0)
while True:
    flag,img = capture.read()
    if flag:
        faces= haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
        face=img [y:y+h, x:x+w, :]
        face=cv2.resize(face,(50,50))
        if(len(data))<12000:
            data.append(face)
            print(len(data))
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27 or len(data)>=12000:
        break    
capture.release()    
cv2.destroyAllWindows()


np.save('withmask.npy',data)