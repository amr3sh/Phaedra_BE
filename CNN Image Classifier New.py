#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from os import listdir, makedirs
from os.path import join, exists, expanduser
 
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


# In[3]:


camera_settings = 'D:/FYP/Dataset/Dataset for Camera Settings'
batch_size = 100
cm_classes = ['c - ISO Low', '1 - Correct', 'b - Shutter Speed Low', 'd - ISO High', 'a - Shutter Speed High']


# In[24]:
# Read Accident Detection Data


cm_iso_low=[]
cm_correct=[]
cm_spd_low=[]
cm_iso_high=[]
cm_spd_high=[]

import glob, os
for indv in cm_classes:
  os.chdir(camera_settings+"/"+indv)
  for file in glob.glob("*.jpg"):
    if indv == 'c - ISO Low':
      image = cv2.imread(camera_settings+"/"+indv+"/"+file)
      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
      cm_iso_low.append(image)
    if indv == '1 - Correct':
      image = cv2.imread(camera_settings+"/"+indv+"/"+file)
      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
      cm_correct.append(image)
    if indv == 'b - Shutter Speed Low':
      image = cv2.imread(camera_settings+"/"+indv+"/"+file)
      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
      cm_spd_low.append(image)
    if indv == 'd - ISO High':
      image = cv2.imread(camera_settings+"/"+indv+"/"+file)
      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
      cm_iso_high.append(image)
    if indv == 'a - Shutter Speed High':
      image = cv2.imread(camera_settings+"/"+indv+"/"+file)
      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
      cm_spd_high.append(image)


# In[25]:


# Generate Labels for each Class in the Data
camera_settings_lbls=[]

for itm in cm_iso_low:
  if itm is not None:
    camera_settings_lbls.append([1,0,0,0,0])
  else:
    print("None")

for itm in cm_correct:
  if itm is not None:
    camera_settings_lbls.append([0,1,0,0,0])
  else:
    print("None")

for itm in cm_spd_low:
  if itm is not None:
    camera_settings_lbls.append([0,0,1,0,0])
  else:
    print("None")

for itm in cm_iso_high:
  if itm is not None:
    camera_settings_lbls.append([0,0,0,1,0])
  else:
    print("None")

for itm in cm_spd_high:
  if itm is not None:
    camera_settings_lbls.append([0,0,0,0,1])
  else:
    print("None")


# In[26]:


# Merge All The Data
# Merge Data
camera_settings_imgs=[]

for itm in cm_iso_low:
  if itm is not None:
    camera_settings_imgs.append(itm)
  else:
    print("None")

for itm in cm_correct:
  if itm is not None:
    camera_settings_imgs.append(itm)
  else:
    print("None")

for itm in cm_spd_low:
  if itm is not None:
    camera_settings_imgs.append(itm)
  else:
    print("None")

for itm in cm_iso_high:
  if itm is not None:
    camera_settings_imgs.append(itm)
  else:
    print("None")

for itm in cm_spd_high:
  if itm is not None:
    camera_settings_imgs.append(itm)
  else:
    print("None")


# In[27]:


camera_settings_imgs = np.asarray(camera_settings_imgs).astype('float32')
camera_settings_lbls = np.asarray(camera_settings_lbls).astype('float32')


# In[28]:


print(len(camera_settings_imgs))
print(len(camera_settings_lbls))


# In[29]:


# Split Data into Train and Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(camera_settings_imgs, camera_settings_lbls, test_size=0.2, random_state=42)


# # Model Building

# ## Base Model

# In[40]:


# Base Model
# base model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
#create model
model_1 = Sequential()
#add model layers
model_1.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(128,128,3)))
model_1.add(Conv2D(32, kernel_size=2, activation='relu'))
model_1.add(Flatten())
model_1.add(Dense(5, activation='softmax'))


# In[41]:


#compile base model using accuracy to measure model performance
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[42]:


# Train Base Model
history_base = model_1.fit(X_train, y_train, epochs=10, shuffle = True, verbose = 1, validation_split=0.2)


# In[45]:


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_base.history['accuracy'], label='Training Accuracy')
plt.plot(history_base.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.ylabel('Accuracy(Training and Validation)')
plt.xlabel('epoch')
plt.legend(loc='lower right')
#plt.legend(['Training'],['Validation'], loc='upper left')
plt.title('Training and Validation Accuracy')

#plt.legend(['Pretrained'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_base.history['loss'], label='Training Loss')
plt.plot(history_base.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.ylabel('Loss(Training and Validation)')
plt.xlabel('epoch')
#plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.legend(loc='upper right')
plt.legend(loc='upper right')
plt.show()


# In[46]:


train_predictions = model_1.predict(X_train)
test_predictions = model_1.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Train Data: "+str(accuracy_score(y_train.argmax(axis=-1), train_predictions.argmax(axis=-1))))
print("Test Data: "+str(accuracy_score(y_test.argmax(axis=-1), test_predictions.argmax(axis=-1))))


# ## ResNet50

# In[47]:


#import inception with pre-trained weights. do not include fully #connected layers
resnet50 = applications.ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = resnet50.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(5, activation='softmax')(x)
# create the full network so we can train on it
resnet50 = Model(inputs=resnet50.input, outputs=predictions)


# In[48]:


# Compile
resnet50.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[49]:


print(X_train.shape)
print(y_train.shape)


# In[51]:


history_resnet50 = resnet50.fit(X_train, y_train, epochs=10, shuffle = True, verbose = 1, validation_split=0.2)


# In[52]:


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_resnet50.history['accuracy'], label='Training Accuracy')
plt.plot(history_resnet50.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.ylabel('Accuracy(Training and Validation)')
plt.xlabel('epoch')
plt.legend(loc='lower right')
#plt.legend(['Training'],['Validation'], loc='upper left')
plt.title('Training and Validation Accuracy')

#plt.legend(['Pretrained'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_resnet50.history['loss'], label='Training Loss')
plt.plot(history_resnet50.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.ylabel('Loss(Training and Validation)')
plt.xlabel('epoch')
#plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.legend(loc='upper right')
plt.legend(loc='upper right')
plt.show()


# In[53]:


train_predictions = resnet50.predict(X_train)
test_predictions = resnet50.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Train Data: "+str(accuracy_score(y_train.argmax(axis=-1), train_predictions.argmax(axis=-1))))
print("Test Data: "+str(accuracy_score(y_test.argmax(axis=-1), test_predictions.argmax(axis=-1))))


# ## ResNet50V2

# In[54]:


# ResNet50V2
resnet50v2 = applications.ResNet50V2(weights='imagenet', include_top=False)

# Add a Global Spatial Average Pooling Layer
x = resnet50v2.output
x = GlobalAveragePooling2D()(x)
# Add a Fully Connected Layer
x = Dense(512, activation='relu')(x)
# And a Fully Connected Output/Classification Layer
predictions = Dense(5, activation='softmax')(x)
# Create the Full Network so we can train on it
resnet50v2 = Model(inputs=resnet50v2.input, outputs=predictions)


# In[55]:


# Compile
resnet50v2.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[56]:


history_resnet50v2 = resnet50v2.fit(X_train, y_train, epochs=5, shuffle = True, verbose = 1, validation_split=0.2)


# In[57]:


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_resnet50v2.history['accuracy'], label='Training Accuracy')
plt.plot(history_resnet50v2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.ylabel('Accuracy(Training and Validation)')
plt.xlabel('epoch')
plt.legend(loc='lower right')
#plt.legend(['Training'],['Validation'], loc='upper left')
plt.title('Training and Validation Accuracy')

#plt.legend(['Pretrained'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_resnet50v2.history['loss'], label='Training Loss')
plt.plot(history_resnet50v2.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.ylabel('Loss(Training and Validation)')
plt.xlabel('epoch')
#plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.legend(loc='upper right')
plt.legend(loc='upper right')
plt.show()


# In[58]:


train_predictions = resnet50v2.predict(X_train)
test_predictions = resnet50v2.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Train Data: "+str(accuracy_score(y_train.argmax(axis=-1), train_predictions.argmax(axis=-1))))
print("Test Data: "+str(accuracy_score(y_test.argmax(axis=-1), test_predictions.argmax(axis=-1))))


# In[ ]:


# Save the Model
resnet50.save("D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/FYP/resnet50.h5")

