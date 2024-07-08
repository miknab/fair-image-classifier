import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D 

from facecls import fcaux

def mlp(num_classes, in_shape, n_hidden = (128, ), activation="relu"):
    """
    Simple multi-layer perceptron
    """
    fcaux.set_seed()
    
    input_img = Input(shape=in_shape)

    dense = Dense(n_hidden[0], activation=activation)(input_img)

    for n_neurons in n_hidden[1:]:
        dense = Dense(n_neurons, activation=activation)(dense)
    
    if num_classes == 2:
        out = Dense(num_classes, activation="sigmoid")(dense)
    elif num_classes > 2:
        out = Dense(num_classes, activation="softmax")(dense)
    else:
        out = Dense(1, activation="linear")(dense)
    
    mlp = Model(input_img, out)

    adam = keras.optimizers.Adam()

    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics_list = ["accuracy"]
        
    elif num_classes > 2:
        loss = "categorical_crossentropy"
        metrics_list = ["accuracy"]
        
    else:
        loss = "mean_squared_error"
        #loss = "mean_absolute_percentage_error"
        metrics_list = ["r2_score", "mean_squared_error", "mean_absolute_percentage_error"]
        
    mlp.compile(optimizer=adam,
                 loss=loss,
                 metrics=metrics_list
                 )

    return mlp

def my_cnn(num_classes, seed=42):
    """
    CNN inspired by the scheme of "AlexNet" as shown in 
    https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f.

    Notice: This is NOT an AlexNet.
    """
    fcaux.set_seed()
    
    input_img = Input(shape=(48, 48 ,1))

    conv1 = Conv2D(16, (3,3), padding="same", strides=(1,1))(input_img)
    activ1 = Activation("relu")(conv1)
    conv2 = Conv2D(32, (5,5), strides=(1,1))(activ1)
    activ2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2,2))(activ2)
    conv3 = Conv2D(64, (5,5), strides=(1,1))(pool1)
    activ3 = Activation("relu")(conv3)
    pool2 = MaxPooling2D(pool_size=(2,2))(activ3)
        
    flat = Flatten()(pool2)
    
    dense1 = Dense(128, activation="relu")(flat) # previously: 128
    
    penultimate = dense1
    
    if num_classes == 2:
        out = Dense(num_classes, activation="sigmoid")(penultimate)
    elif num_classes > 2:
        out = Dense(num_classes, activation="softmax")(penultimate)
    else:
        out = Dense(1, activation="linear")(penultimate)

    model = Model(input_img, out)

    adam = keras.optimizers.Adam()

    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics_list = ["accuracy"]
        
    elif num_classes > 2:
        loss = "categorical_crossentropy"
        metrics_list = ["accuracy"]
        
    else:
        loss = "mean_squared_error"
        #loss = "mean_absolute_percentage_error"
        metrics_list = ["r2_score", "mean_squared_error", "mean_absolute_percentage_error"]
        
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=metrics_list
                 )

    return model

def alex_net(num_classes):
    """
    Creates an AlexNet-like CNN.

    Notice: There is only one dense layer between the Flatten()
    and the output layer to reduce the size of the model.
    """
    fcaux.set_seed()

    input_img = Input(shape=(newdim,newdim,1))

    conv1 = Conv2D(96, (11,11), strides=4)(input_img)
    activ1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(3,3), strides=2)(activ1)

    conv2 = Conv2D(256, (5,5), padding="same")(pool1)
    activ2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(3,3), strides=2)(activ2)
    
    conv3 = Conv2D(384, (3,3), padding="same")(pool2)
    activ3 = Activation("relu")(conv3)
    conv4 = Conv2D(384, (3,3), padding="same")(activ3)
    activ4 = Activation("relu")(conv4)
    conv5 = Conv2D(256, (3,3), padding="same")(activ4)
    activ5 = Activation("relu")(conv4)
    pool3 = MaxPooling2D(pool_size=(3,3), strides=2)(activ2)
    
    flat = Flatten()(pool3)
    
    dense1 = Dense(4096, activation="relu")(flat) # previously: 128
    drop1 = Dropout(0.5)(dense1)
    #dense2 = Dense(2048, activation="relu")(drop1)
    #drop2 = Dropout(0.5)(dense2)

    penultimate = drop1
    
    if num_classes == 2:
        out = Dense(num_classes, activation="sigmoid")(penultimate)
    elif num_classes > 2:
        out = Dense(num_classes, activation="softmax")(penultimate)
    else:
        out = Dense(1, activation="linear")(penultimate)

    model = Model(input_img, out)

    adam = keras.optimizers.Adam()

    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics_list = ["accuracy"]
        
    elif num_classes > 2:
        loss = "categorical_crossentropy"
        metrics_list = ["accuracy"]
        
    else:
        loss = "mean_squared_error"
        #loss = "mean_absolute_percentage_error"
        metrics_list = ["r2_score", "mean_squared_error", "mean_absolute_percentage_error"]
        
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=metrics_list
                 )

    return model