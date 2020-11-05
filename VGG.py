from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Activation

custom_vgg = Sequential()
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = (32, 32, 3)))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Flatten())
custom_vgg.add(Dense(10, activation = "softmax"))

custom_vgg.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

custom_vgg.summary()


# Output params summary 
# =================================================================
# Total params: 307,498
# Trainable params: 307,498
# Non-trainable params: 0