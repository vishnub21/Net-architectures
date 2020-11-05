
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Activation


def dense_layer(x, layer_configs):
  layers = []
  for i in range(2):
    if layer_configs[i]["layer_type"] == "Conv2D":
        layer = Conv2D(layer_configs[i]["filters"], layer_configs[i]["kernel_size"], strides = layer_configs[i]["strides"], padding = layer_configs[i]["padding"], activation = layer_configs[i]["activation"])(x)
    layers.append(layer)
  for n in range(2, len(layer_configs)):
    if layer_configs[n]["layer_type"] == "Conv2D":
      layer = Conv2D(layer_configs[n]["filters"], layer_configs[n]["kernel_size"], strides = layer_configs[n]["strides"], padding = layer_configs[n]["padding"], activation = layer_configs[n]["activation"])(concatenate(layers, axis = 3))
    layers.append(layer)
  return layer

# -- 

layer_f8 = [
    {
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

layer_f16 = [
    {
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

layer_f32 = [
    {
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

layer_f64 = [
    {
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

layer_f128 = [
    {
        "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
        "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
]

#Custom DenseNet
inp = Input(shape = (32, 32, 3))
x = inp
x = Conv2D(4, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = dense_layer(x, layer_f8)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f16)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f32)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f64)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f128)
x = Dropout(0.8)(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization(axis = 3)(x)
x = Conv2D(96, (1, 1), activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)

x = MaxPooling2D((2, 2))(x)
x = BatchNormalization(axis = 3)(x)
x = Flatten()(x)

x = Dropout(0.4)(x)
x = Dense(14, activation = "softmax")(x)

dense_net = Model(inp, x)
dense_net.summary()

#Output params summary
#==================================================================================================
#Total params: 2,065,686
#Trainable params: 2,064,806
#Non-trainable params: 880