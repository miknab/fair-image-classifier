# Standard library imports
from typing import Tuple

# 3rd party imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.nn import local_response_normalization

# Local imports
from facecls import fcaux

def mlp(num_classes: int, 
        in_shape: Tuple[int], 
        n_hidden: Tuple[int]=(128, ), 
        activation: str="relu",
        seed: int=42
       ) -> keras.src.models.functional.Functional:
    """
    Define and create a simple multi-layer perceptron.

    This function defines, creates and returns a simple
    multi-layer perceptron with an architecture defined
    by the input parameters. The optimizer is hard-coded
    to be an ADAM optimizer.

    Notes
    -----
    This function assumes that there is at least one
    hidden layer.

    Parameters
    ----------
    num_classes : int
        Number of classes in case of classification (e.g.
        num_classes = 2 for binary classification).
        
    in_shape : Tuple[int]
        Number of features/neurons in input layer.

    n_hidden : Tuple[int]
        Number of hidden layers (length of the tuple) and 
        number of neurons per hidden layer (value of elements
        in the tuple).
        default: (128, )

    activation : str
        Activation function used in all neurons but those in
        the output layer.

    seed : int
        value of the seed to be used
        default: 42

    Returns
    -------
    keras.src.models.functional.Functional
        compiled tensorflow.keras model containing the multi-
        layer perceptron.

    Raises
    ------
    ValueError
        If len(n_hidden)<1.
    """
    # First check if the assumption is satisfied. If not, throw
    # an error
    if len(n_hidden)<1:
        raise ValueError("MLPs without any hidden layers are not allowed. "
                         + "len(n_hidden) must be >=1."
                        )
        
    # Now that we know the definition of the MLP satisfies the
    # assumptions, set the seed for reproducible results
    fcaux.set_seed(seed)

    # In order to define the MLP, we use the functional API in
    # TensorFlow.
    # -- Start by defining the input object
    input_img = Input(shape=in_shape)

    # -- Now add the first hidden layer taking the previously
    #    defined input object as input
    dense = Dense(n_hidden[0], activation=activation)(input_img)

    # -- Iteratively add more hidden layers if required
    for n_neurons in n_hidden[1:]:
        dense = Dense(n_neurons, activation=activation)(dense)

    # -- Define the output layers according to num_classes.
    #    Notice that the activation function of the output
    #    neuron depends on the value of num_classes
    if num_classes == 2:
        out = Dense(num_classes, activation="sigmoid")(dense)
    elif num_classes > 2:
        out = Dense(num_classes, activation="softmax")(dense)
    else:
        out = Dense(1, activation="linear")(dense)

    # Put the model together, ...
    mlp = Model(input_img, out)

    # ... define the optimizer, the loss, and the metrics ...
    adam = keras.optimizers.Adam()

    if num_classes == 2:
        # Binary classification
        loss = "binary_crossentropy"
        metrics_list = ["accuracy"]
        
    elif num_classes > 2:
        # Multi-class classification
        loss = "categorical_crossentropy"
        metrics_list = ["accuracy"]
        
    else:
        # Regression
        loss = "mean_squared_error"
        metrics_list = ["r2_score", "mean_squared_error", "mean_absolute_percentage_error"]

    # ... and finally compile the model
    mlp.compile(optimizer=adam,
                loss=loss,
                metrics=metrics_list
               )

    # Return the compiled model
    return mlp

def my_cnn(num_classes: int, seed: int = 42) -> keras.src.models.functional.Functional:
    """
    Definition of an AlexNet-inspired CNN.
    
    This function defines a CNN. While it is not an 
    AlexNet, it is somewhat inspired by it. Specifically,
    it follows the scheme of "AlexNet" as shown in 
    https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f.

    Parameters
    ----------
    num_classes : int
        Number of classes in case of classification (e.g.
        num_classes = 2 for binary classification).

    seed : int
        value of the seed to be used
        default: 42
        
    Returns
    -------
    keras.src.models.functional.Functional
        compiled tensorflow.keras model containing the CNN.
    """
    # Start again by setting the seed
    fcaux.set_seed(seed)

    # Next, again define the input object. Here the object
    # has three channels as we are processing images. However,
    # as the images are gray scale, the dimension of the 3rd
    # channel is 1.
    input_img = Input(shape=(48, 48 ,1))

    # Define the CNN architecture: 
    conv1 = Conv2D(16, (3,3), padding="same", strides=(1,1))(input_img)
    activ1 = Activation("relu")(conv1)
    conv2 = Conv2D(32, (5,5), strides=(1,1))(activ1)
    activ2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2,2))(activ2)
    conv3 = Conv2D(64, (5,5), strides=(1,1))(pool1)
    activ3 = Activation("relu")(conv3)
    pool2 = MaxPooling2D(pool_size=(2,2))(activ3)

    # Flattening layer acting as interface between CNN and 
    # dense NN for classification/regression
    flat = Flatten()(pool2)

    # Define architecture of dense NN for classification/regression
    dense1 = Dense(128, activation="relu")(flat)
    penultimate = dense1

    # Output layer for classification/regression
    if num_classes == 2:
        # Binary classification
        out = Dense(num_classes, activation="sigmoid")(penultimate)
    elif num_classes > 2:
        # Multi-class classification
        out = Dense(num_classes, activation="softmax")(penultimate)
    else:
        # Regression
        out = Dense(1, activation="linear")(penultimate)

    # Put the model together, ...
    model = Model(input_img, out)

    # ... define the optimizer, the loss, and the metrics ...
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

    # ... and finally compile the model
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=metrics_list
                 )

    # Return the model
    return model

def alex_net(num_classes: int, newdim: int, seed: int=42) -> keras.src.models.functional.Functional:
    """
    Creates an AlexNet-like CNN.

    This function returns a CNN that is almost an AlexNet as
    described in Krizhevsky, Sutskever and Hinton (2012). The
    network defined in this function deviates from a proper
    AlexNet in the following four aspects:

    1) The input dimension can but does not have to be the equal
       to 224x224
    2) Here we focus on gray-scale instead of colored images, i.e.
       we use single-channel images as inputs.
    3) In order to make the model computationally less expensive,
       there is only one dense layer between the flattening and
       the output layer.
    4) The output dimension can be any positive number (originally
       it was 1000 to solve the ImageNet LSVRC-2010 contest).

    Parameters
    ----------
    num_classes : int
        Number of classes in case of classification (e.g.
        num_classes = 2 for binary classification).

    newdim : int
        Physical dimension of input image (used for both
        height and width)

    seed : int
        value of the seed to be used
        default: 42
        
    Returns
    -------
    keras.src.models.functional.Functional
        compiled tensorflow.keras model containing the CNN.

    Raises
    ------
    ValueError
        If newdim < 1
    """
    # Check sanity of inputs
    if newdim < 1:
        raise ValueError("newdim must be at least 1.")
        
    # Start again by setting the seed
    fcaux.set_seed(seed)

    # Before we start constructing the actual CNN, let's define
    # the local response-normalization layer, which will be used
    # twice in the code below:
    lrn = Lambda(local_response_normalization(bias=2, 
                                              alpha=1e-4, 
                                              beta=0.75
                                              depth_radius=5)
                                             )
    
    # Next, we again define the input object. Here the object
    # has three channels as we are processing images. However,
    # as the images are gray scale, the dimension of the 3rd
    # channel is 1.
    input_img = Input(shape=(newdim,newdim,1))

    # Then the AlexNet architecture is defined as follows:
    conv1 = Conv2D(96, (11,11), strides=4)(input_img)
    activ1 = Activation("relu")(conv1)
    norm1 = lrn(activ1)
    pool1 = MaxPooling2D(pool_size=(3,3), strides=2)(norm1)

    conv2 = Conv2D(256, (5,5), padding="same")(pool1)
    activ2 = Activation("relu")(conv2)
    norm2 = lrn(activ2)
    pool2 = MaxPooling2D(pool_size=(3,3), strides=2)(norm2)
    
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

    penultimate = drop1

    # Add the output layer whose dimension depends on the task at hand
    if num_classes == 2:
        out = Dense(num_classes, activation="sigmoid")(penultimate)
    elif num_classes > 2:
        out = Dense(num_classes, activation="softmax")(penultimate)
    else:
        out = Dense(1, activation="linear")(penultimate)

    # Put everything together, ...
    model = Model(input_img, out)

    # ... define the optimizer, the loss, and the metrics ...
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

    # ... and finally compile the model.
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=metrics_list
                 )

    # return the AlexNet model
    return model