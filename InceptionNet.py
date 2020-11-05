from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Dropout, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, Activation


def inception_layer(x, layer_configs):
  layers = []
  for configs in layer_configs:
    if configs[0]["layer_type"] == "Conv2D":
      layer = Conv2D(configs[0]["filters"], configs[0]["kernel_size"], strides = configs[0]["strides"], padding = configs[0]["padding"], activation = configs[0]["activation"])(x)
    if configs[0]["layer_type"] == "MaxPooling2D":
      layer = MaxPooling2D(configs[0]["kernel_size"], strides = configs[0]["strides"], padding = configs[0]["padding"])(x)
    for n in range(1, len(configs)):
      if configs[n]["layer_type"] == "Conv2D":
        layer = Conv2D(configs[n]["filters"], configs[n]["kernel_size"], strides = configs[n]["strides"], padding = configs[n]["padding"], activation = configs[n]["activation"])(layer)
      if configs[n]["layer_type"] == "MaxPooling2D":

        layer = MaxPooling2D(configs[n]["kernel_size"], strides = configs[n]["strides"], padding = configs[n]["padding"])(layer)
    layers.append(layer)
  return concatenate(layers, axis = 3)
  

layer_3a = [[{
    "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (1, 1), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D", "filters" : 96, "kernel_size" : (1, 1), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
    "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (1, 1), "strides" : 1, "padding" : "same", "activation" : "relu"
    },{
    "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (5, 5), "strides" : 1, "padding" : "same", "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 32,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]

layer_3b = [[{
    "layer_type" : "Conv2D",
    "filters" : 128,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 128,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 192,
    "kernel_size" : (3, 3),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 32,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 96,
    "kernel_size" : (5, 5),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 96,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]

layer_4a = [[{
    "layer_type" : "Conv2D",
    "filters" : 192,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 96,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 208,
    "kernel_size" : (3, 3),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 16,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 48,
    "kernel_size" : (5, 5),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 64,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]

layer_4b = [[{
    "layer_type" : "Conv2D",
    "filters" : 160,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 112,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 224,
    "kernel_size" : (3, 3),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
    "layer_type" : "Conv2D",
    "filters" : 24,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    },{
    "layer_type" : "Conv2D",
    "filters" : 64,
    "kernel_size" : (5, 5),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
],[
    {
        "layer_type" : "MaxPooling2D",
        "strides" : 1,
        "kernel_size" : (3, 3),
        "padding" : "same"
    },{
    "layer_type" : "Conv2D",
    "filters" : 64,
    "kernel_size" : (1, 1),
    "strides" : 1,
    "padding" : "same",
    "activation" : "relu"
    }
]]


inp = Input(shape = (32, 32, 3))
x = inp
x = Conv2D(64, (7, 7), strides = 2, padding = "same", activation = "relu")(x)
x = MaxPooling2D((3, 3), padding = "same", strides = 2)(x)
x = Conv2D(64, (1, 1), strides = 1, padding = "same", activation = "relu")(x)
x = Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = MaxPooling2D((3, 3), padding = "same", strides = 2)(x)
x = inception_layer(x, layer_3a)
x = inception_layer(x, layer_3b)
x = MaxPooling2D((3, 3), padding = "same", strides = 2)(x)
x = inception_layer(x, layer_4a)

x1 = AveragePooling2D((2, 2), strides = 3)(x)
x1 = Conv2D(128, (1, 1), padding = "same", activation = "relu")(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation = "relu")(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(100, activation = "softmax")(x1)

inc = Model(inp, x1)
inc.summary()

#Output Params Summary
#==================================================================================================
#Total params: 1,373,284
#Trainable params: 1,373,284