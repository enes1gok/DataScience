import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.model_selection import train_test_split

data = pd.read_excel("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/DeepLearning/DLproject/date_fruit.xlsx")
'''print(data.head())
print(data.shape)
print(data['Class'].unique())'''
#for better understanding of our data, let's split the dataset into features and labels
# create x and y dataset using drop() and loc() methods

x = data.drop(['Class'], axis=1)
y = data.loc[:, ['Class']]

#min_max scale = NORMALIZATION

x_scaled = minmax_scale(x)

x = pd.DataFrame(x_scaled)

'''print(x.head())
print(y)'''

#AI algorithms can not understand string datas so we should convert them

encoder = LabelEncoder()
y = encoder.fit_transform(y)
#print(y)

x_train, x_temporary, y_train, y_temporary = train_test_split(x, y, train_size=0.8)
x_val, x_test, y_val, y_test = train_test_split(x_temporary, y_temporary, train_size=0.5)

'''print(len(x))
print(len(y))
print(len(x_val))
print(len(y_val))'''


#CONSTRUCTING NEURAL NETWORK

import tensorflow as tf

model = tf.keras.Sequential()

#create an input layer

input_layer = tf.keras.layers.Dense(4096, input_shape=(34,), activation = 'relu')

model.add(input_layer)

#add hidden layers

model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))

#add output layer
model.add(tf.keras.layers.Dense(7, activation = 'softmax'))

#configure the model for training

model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Training the model

results = model.fit(x_train, y_train, epochs = 100, validation_data = (x_val, y_val))

plt.plot(results.history['loss'],label = 'Train')

plt.plot(results.history['val_loss'],label = 'Test')

plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

test_result = model.test_on_batch(x_test, y_test)
print(test_result) #first element is result of the loss function(should be lowest) and second element is the accuracy(should be highest)