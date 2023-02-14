import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/DeepLearning/train.csv")
#print(data.shape) (42.000,785)

#Let's separate the labels.
Y_data = data['label']

#Let's create x train dataset.
X_data = data.drop(['label'],axis=1)

#visualize number of digits classes
'''plt.figure(figsize=(15,7))
g = sns.countplot(Y_data, palette = 'icefire')
plt.title('Classes')
plt.show()'''
#print("Number of classes and samples: ", Y_data.value_counts())

#we will use the size of the image.
image_size = int(np.sqrt(X_data.shape[1]))
#print(image_size)

#let's visualize our data 3 class
'''image1 = X_data.iloc[2000].values
image1 = image1.reshape((image_size,image_size))
plt.imshow(image1, cmap='gray')
plt.axis('off')
plt.show()'''
#let's visualize our data 4 class
'''image2 = X_data.iloc[19000].values
image2 = image2.reshape((image_size,image_size))
plt.imshow(image2, cmap='gray')
plt.axis('off')
plt.show()'''

#normalize the data
X_data = X_data / 255.0

#reshape
X_data = X_data.values.reshape(-1,28,28,1)
#print("X data size: ",X_data.shape)

#label coding
from keras.utils.np_utils import to_categorical #to convert the vector
Y_data = to_categorical(Y_data, num_classes = 10)

#TRAIN TEST DIVISION
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=2)

'''print("X Train size:",X_train.shape)
print("X Test size:",X_test.shape)
print("Y Train size:",Y_train.shape)
print("Y Test size:",Y_test.shape)'''


from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (5,5), padding='Same', activation='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (5,5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

#full link
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#We will use the ADAM optimization method.
optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)

#We will use the categorical cross-entropy cost method.

#Let's compile the model.
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Suppose we have a dataset with 10 samples.
#Let batch_size = 2.
##Let epochs = 3.
#We have 5 batches for each epoch (10/2=5).
#10 pieces of data complete the cycle 5 times, 2 in each epoch.
epochs = 10
batch_size = 250

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,#girdi ortalamasını veri kümesi üzerinden 0 olarak ayarlayın.
    samplewise_center=False,#her bir örnek ortalamasını 0 olarak ayarlayın.
    featurewise_std_normalization=False,#girdileri veri kümesinin standartlarına böl
    samplewise_std_normalization=False,#her girdiyi std'ye böl
    zca_whitening=False,#dimension reduction
    rotation_range=5,#döndürme yapayım mı
    zoom_range=0.1,# 10% görüntüyü rastgele yakınlaştır.
    width_shift_range=0.1,# görüntüleri yatay olarak rastgele kaydır 10%
    height_shift_range=0.1,# görüntüleri dikey olarak rastgele kaydır 10%
    horizontal_flip=False,#görüntüleri yatay olarak rastgele çevir.
    vertical_flip=False)#görüntüleri dikey olarak rastgele çevir.
datagen.fit(X_train)

#Training the model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs = epochs, validation_data=(X_test, Y_test), steps_per_epoch=X_train.shape[0] // batch_size, verbose=2)

#evolution of the model
'''plt.plot(history.history['val_loss'], color = 'b')
plt.title("Test Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()'''

#predict from data test set
Y_prediction = model.predict(X_test)

#convert the predicted data.
Y_prediction_classes = np.argmax(Y_prediction, axis = 1)

#convert the test data
Y_true = np.argmax(Y_test, axis = 1)

#calculate confusion matrix
confusion_mtx = confusion_matrix(Y_true,Y_prediction_classes) 

#ploting confusion matrix
f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx, annot = True, linewidths=0.01, cmap="Greens", linecolor="r")
plt.xlabel("Predicted label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
