
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Activation


stride = 1
CHANNEL_AXIS = 3

def res_layer(x ,filters,pooling = False,dropout = 0.0):
    temp = x
    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)
    temp = BatchNormalization(axis = CHANNEL_AXIS)(temp)
    temp = Activation("relu")(temp)
    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)

    x = add([temp,Conv2D(filters,(3,3),strides = stride,padding = "same")(x)])
    if pooling:
        x = MaxPooling2D((2,2))(x)
    if dropout != 0.0:
        x = Dropout(dropout)(x)
    x = BatchNormalization(axis = CHANNEL_AXIS)(x)
    x = Activation("relu")(x)
    return x

# Creating Custom ResNet 
inp = Input(shape = (32,32,3))
x = inp
x = Conv2D(16,(3,3),strides = stride,padding = "same")(x)
x = BatchNormalization(axis = CHANNEL_AXIS)(x)
x = Activation("relu")(x)
x = res_layer(x,32,dropout = 0.2)
x = res_layer(x,32,dropout = 0.3)
x = res_layer(x,32,dropout = 0.4,pooling = True)
x = res_layer(x,64,dropout = 0.2)
x = res_layer(x,64,dropout = 0.2,pooling = True)
x = res_layer(x,256,dropout = 0.4)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(4096,activation = "relu")(x)
x = Dropout(0.23)(x)
x = Dense(100,activation = "softmax")(x)

resnet_model = Model(inp,x,name = "Resnet")
resnet_model.summary()

#output params summary
# ==================================================================================================
# Total params: 68,671,236
# Trainable params: 68,669,284
# Non-trainable params: 1,952