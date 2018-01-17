import keras
import numpy as np
from keras.layers.advanced_activations import *
from keras.callbacks import *
from keras.datasets import *
from keras.layers import *
from keras.metrics import *
from keras.models import *
from keras.optimizers import *

#######################################################

experiment_name = "CIFAR_10_E500_D512_C16.3.3_Lr0.01_Relu"

# chargement du dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#Reshape de nos données pour avoir la valeurs de chaque pixel

x_train = np.reshape(x_train, (-1, 32, 32, 3)) / 255.0
x_test = np.reshape(x_test, (-1, 32, 32, 3)) / 255.0

print(x_train.shape)
print(y_train.shape)

########################################################

tb_callback = TensorBoard("./logs/" + experiment_name, )

########################################################

dim = 32*32*3
dense = 10

model = Sequential()


print("Model training will start soon")
# model.add(Conv2D(48, (3, 3), padding='same', input_shape=(3, 32, 32)))
# model.add(MaxPool2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(dense, activation=tanh))


model.add(Conv2D(16, (3, 3), padding='same',  activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(16, (3, 3), padding='same',  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

###############################################
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

###############################################
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy',metrics=[categorical_accuracy], optimizer=sgd)
model.compile(sgd, mse, metrics=[categorical_accuracy])


# model.fit(x_train, y_train, batch_size=32, epochs=1, callbacks=[tb_callback])
model.fit(x_train, y_train, batch_size=256, epochs=500, callbacks=[tb_callback], validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size=128)
print(score)