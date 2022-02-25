#python -m venv C:/Documents/2021 dev projects/MLImages to create env
#source c:/Users/gavin/Documents/2021_dev_projects/MLImages/dev/Scripts/activate
#in Conda prompt type: python classifier.py to run with conda
#ll
#to load and save models: https://www.youtube.com/watch?v=idus3KO6Wic&ab_channel=AladdinPersson

#The model we are designing is a Convolutional neural network used for image classification

#tensorflow dependancies
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

#functional dependencies
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize



plt.style.use('fivethirtyeight')

print(tf.__version__)
from keras.datasets import cifar10
#import dataset from cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#lets look at the data types of the variables
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

'''output:
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>'''

#get the shape of the arrays
print('x_train shape:', x_train.shape) 
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

'''output:
x_train shape: (50000, 32, 32, 3) -> contains 50000 rows of data that are 32x32 images with depth 3 (RGB)
y_train shape: (50000, 1) -> contains 50000 rows of data and 1 column so its 2 dimensional
x_test shape: (10000, 32, 32, 3) -> contains 10000 rows of data that are 32x32 images with depth 3 (RGB)
y_test shape: (10000, 1) -> contains 10000 rows of data and 1 column so its 2 dimensional

x_train is the image data, y_train is the image labels, same for test data
'''

#lets take a look at the first image as an array
index = 10

print(x_train[index])

#lets how image as a picture 

#img = plt.imshow(x_train[index])
#plt.show()

#grab image label
print('image label is: ', y_train[index]) # output: [6] -> list with one element in it
#

#get image classification
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse', 'ship','truck']
#print the image class

print('image class is: ', classification[y_train[index][0]]) #we want the

#Convert the labels into a set of 10 numbers to input into the neural network

y_train_one_hot = to_categorical(y_train) #converts integers into a matrix of values between 0 and 1
y_test_one_hot = to_categorical(y_test)

#print the labels
print(y_test_one_hot)

'''
1 corresponds to the category that the particular datapoint is from
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]]
'''

#print the new label of the current image/picture above 
print('the one_hot_label is: ',y_train_one_hot[index])

#Normalize the pixels to be values between 0 and 1
x_train = x_train/255
x_test = x_test/255

print(x_train[index])

#Create model architecture
model = Sequential()
#Add first layer wwhich will be a convolutional layer to extract features from the image
#will create 32, 5x5 'relu' convoluted features (feature maps)
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3)))

#layer 2: pooling layer: reduces the dimensions of the feature map thus reducing the number of parameters
#Max pooling operation selects the maximum element in a region (defined by pool_size) of the feature map
#Output will contain the most prominent features of the previous feature map
#2x2 pixel filter to attain the max element from the feature maps
model.add(MaxPooling2D(pool_size=(2,2)))

#2nd convolution layer (Hidden layer)
model.add(Conv2D(32,(5,5),activation='relu'))

#add another pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#add a flatten
#converts data to a 1D array
model.add(Flatten())

#add a layer with 1000 neurons
model.add(Dense(1000,activation='relu'))

#add a dropout layer
model.add(Dropout(0.5))

#add a layer with 500 neurons
model.add(Dense(500,activation='relu'))

#add a dropout layer
model.add(Dropout(0.5))

#add a layer with 250 neurons
model.add(Dense(250,activation='relu'))

#add a layer with 10 neurons since we have 10 classes of items
model.add(Dense(10,activation='softmax'))

#Lets compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#Now we train the model
hist = model.fit(x_train,y_train_one_hot,batch_size=64,epochs=30,validation_split=0.2) #lower batch size

print('The model tested against test dataset is')
#Evaluate the model with the test dataset
model.evaluate(x_test,y_test_one_hot)[1]

#Visualize model accuracy and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper right')
#plt.show()


#Visualize model loss and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper left')
#plt.show()

#lets test the model with an example

#grab image
#testImage = Image.open('images/kitty.jpg')
new_image = plt.imread('images/kitty.jpg')
imgg = plt.imshow(new_image)
#plt.show()


#Resize the image to be 32x32 pixels
resized_image = resize(new_image,(32,32,3)) #(32,32,3) is the new output shape
img_resized = plt.imshow(resized_image) 
#plt.show() #display resized

#Now we test the image against our model
predictions = model.predict(np.array([resized_image]))
#print(predictions)

'''raw output:
[[6.6811512e-03 1.5956951e-04 9.4484404e-02 2.7172318e-01 3.4071881e-01
  1.7414337e-01 1.0700769e-01 3.1045293e-03 1.8723563e-03 1.0491974e-04]]
an array of 10 numbers representing the confidence that the model has about each category
'''
#now we sort the predictions from least to greatest
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp


print(list_index)

#print the first five most likely classes
for i in range(5):
    print(classification[list_index[i]], ': ', round(predictions[0][list_index[i]] * 100,2), '%')




print('done!')
print('model is finished')