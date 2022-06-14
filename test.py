from json import load
from tkinter import font
from cv2 import imwrite
from matplotlib import image
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

haar_data=cv2.CascadeClassifier(r"D:\DAA Project\data.xml")

with_mask= np.load(r"D:\DAA Project\withmask.npy")
without_mask= np.load(r"D:\DAA Project\withoutmask.npy")

with_mask=with_mask.reshape(922,50*50*3)
without_mask=without_mask.reshape(1129,50*50*3)

X=np.r_[with_mask,without_mask]

labels = np.zeros(X.shape[0])
labels[922:]=1.0
names={0:'Mask',1.0:'No Mask'}

x_train, x_test, y_train, y_test=train_test_split(X, labels, test_size=0.20)

print(x_train.shape)

pca = PCA(n_components=3)
x_train=pca.fit_transform(x_train)

svm=SVC()
svm.fit(x_train, y_train)

x_test=pca.transform(x_test)


y_pred=svm.predict(x_test)
print(accuracy_score(y_test,y_pred))

img_counter = 0

path=r"D:\DAA Project\new"

capture=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img = capture.read()
    if flag:
        faces= haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
        face=img [y:y+h, x:x+w, :]
        face=cv2.resize(face,(50,50))
        face=face.reshape(1,-1)
        face=pca.transform(face)
        pred=svm.predict(face)[0]
        n=names[int(pred)]
        cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
        if(n=='No Mask'):
            img_name = "{}.png".format(img_counter)
            cv2.imwrite(os.path.join(path,img_name), img)
            img_counter += 1
    cv2.imshow('result',img)
    if cv2.waitKey(100)==27:
        break
    
             
capture.release()    
cv2.destroyAllWindows()