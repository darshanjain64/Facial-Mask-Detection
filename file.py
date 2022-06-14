from dataclasses import dataclass
from imp import load_module
from json import load
from pyexpat import model
from tkinter import font
from cv2 import imwrite
from matplotlib import axis, image
import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

haar_data=cv2.CascadeClassifier(r"D:\DAA Project\data.xml")

data=[]
target=[]

data_path=r"D:\DAA Project\dataset-20220423T022229Z-001\dataset"
categories=os.listdir(data_path)
labels=[i for i in range (len(categories))]

label_dict=dict(zip(categories,labels))

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)

    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
            resized=cv2.resize(gray,(100,100))
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print('Exception:',e)    


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = np_utils.to_categorical(labels)
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],100,100,1))
target=np.array(target)







np.save('123',data)
np.save('target',target)

data = np.load('123.npy')
target = np.load('target.npy')
print(data.shape)
print(target.shape)
model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(200,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

model=load_module('model-017.model')

labels_dict={0:'Mask',1.0:'No Mask'}
color_dict={0:(0,255,0),1.0:(0,0,255)}

img_counter = 0

path=r"D:\DAA Project\new"

capture=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img = capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=haar_data.detectMultiScale(gray,1.3,5)
    
    for x,y,w,h in faces:
        face_img=gray[y:y+h,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))

        result=model.predict(reshaped)
        
        label=np.argmax(result,axis-1)[0]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img,labels_dict[label],(x,y-10),font,1,(244,250,250),2)
        if(labels_dict[label]=='No Mask'):
            img_name = "{}.png".format(img_counter)
            cv2.imwrite(os.path.join(path,img_name), img)
            img_counter += 1
    cv2.imshow('result',img)
    if cv2.waitKey(100)==27:
        break
    
             
capture.release()    
cv2.destroyAllWindows()