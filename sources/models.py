import os
import sys
import settings

# Try to mute and then load Tensorflow and Keras
# Muting seems to not work lately on Linux in any way
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten
sys.stdin = stdin
sys.stderr = stderr

MODEL_NAME_PREFIX = ''

# ---
# Models

# Xception model
def model_base_Xception(input_shape):
    model = Xception(weights=None, include_top=False, input_shape=input_shape)

    # Grab last model layer and attach global average pooling layer
    x = model.output
    x = GlobalAveragePooling2D()(x)

    return model.input, x

# First small CNN model
def model_base_test_CNN(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())

    return model.input, model.output

# 64x3 model
def model_base_64x3_CNN(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Flatten())

    return model.input, model.output

# 4 CNN layer model
def model_base_4_CNN(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())

    return model.input, model.output

# 5 CNN layer with residual connections model
def model_base_5_residual_CNN(input_shape):
    input = Input(shape=input_shape)

    cnn_1 = Conv2D(64, (7, 7), padding='same')(input)
    cnn_1a = Activation('relu')(cnn_1)
    cnn_1c = Concatenate()([cnn_1a, input])
    cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1c)

    cnn_2 = Conv2D(64, (5, 5), padding='same')(cnn_1ap)
    cnn_2a = Activation('relu')(cnn_2)
    cnn_2c = Concatenate()([cnn_2a, cnn_1ap])
    cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2c)

    cnn_3 = Conv2D(128, (5, 5), padding='same')(cnn_2ap)
    cnn_3a = Activation('relu')(cnn_3)
    cnn_3c = Concatenate()([cnn_3a, cnn_2ap])
    cnn_3ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_3c)

    cnn_4 = Conv2D(256, (5, 5), padding='same')(cnn_3ap)
    cnn_4a = Activation('relu')(cnn_4)
    cnn_4c = Concatenate()([cnn_4a, cnn_3ap])
    cnn_4ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_4c)

    cnn_5 = Conv2D(512, (3, 3), padding='same')(cnn_4ap)
    cnn_5a = Activation('relu')(cnn_5)
    #cnn_5c = Concatenate()([cnn_5a, cnn_4ap])
    cnn_5ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_5a)

    flatten = Flatten()(cnn_5ap)

    return input, flatten

# 5 CNN layer with residual connections and no activations model
def model_base_5_residual_CNN_noact(input_shape):
    input = Input(shape=input_shape)

    cnn_1 = Conv2D(64, (7, 7), padding='same')(input)
    #cnn_1a = Activation('relu')(cnn_1)
    cnn_1c = Concatenate()([cnn_1, input])
    cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1c)

    cnn_2 = Conv2D(64, (5, 5), padding='same')(cnn_1ap)
    #cnn_2a = Activation('relu')(cnn_2)
    cnn_2c = Concatenate()([cnn_2, cnn_1ap])
    cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2c)

    cnn_3 = Conv2D(128, (5, 5), padding='same')(cnn_2ap)
    #cnn_3a = Activation('relu')(cnn_3)
    cnn_3c = Concatenate()([cnn_3, cnn_2ap])
    cnn_3ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_3c)

    cnn_4 = Conv2D(256, (5, 5), padding='same')(cnn_3ap)
    #cnn_4a = Activation('relu')(cnn_4)
    cnn_4c = Concatenate()([cnn_4, cnn_3ap])
    cnn_4ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_4c)

    cnn_5 = Conv2D(512, (3, 3), padding='same')(cnn_4ap)
    #cnn_5a = Activation('relu')(cnn_5)
    #cnn_5c = Concatenate()([cnn_5a, cnn_4ap])
    cnn_5ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_5)

    flatten = Flatten()(cnn_5ap)

    return input, flatten

# 5 CNN layer with residual connections model
def model_base_5_wide_CNN(input_shape):
    input = Input(shape=input_shape)

    cnn_1_c1 = Conv2D(64, (7, 7), strides=(3, 3), padding='same')(input)
    cnn_1_a = Activation('relu')(cnn_1_c1)

    cnn_2_c1 = Conv2D(64, (5, 5), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_a1 = Activation('relu')(cnn_2_c1)
    cnn_2_c2 = Conv2D(64, (3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_a2 = Activation('relu')(cnn_2_c2)
    cnn_2_ap = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_c = Concatenate()([cnn_2_a1, cnn_2_a2, cnn_2_ap])

    cnn_3_c1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_a1 = Activation('relu')(cnn_3_c1)
    cnn_3_c2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_a2 = Activation('relu')(cnn_3_c2)
    cnn_3_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_c = Concatenate()([cnn_3_a1, cnn_3_a2, cnn_3_ap])

    cnn_4_c1 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_a1 = Activation('relu')(cnn_4_c1)
    cnn_4_c2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_a2 = Activation('relu')(cnn_4_c2)
    cnn_4_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_c = Concatenate()([cnn_4_a1, cnn_4_a2, cnn_4_ap])

    cnn_5_c1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(cnn_4_c)
    cnn_5_a1 = Activation('relu')(cnn_5_c1)
    cnn_5_gap = GlobalAveragePooling2D()(cnn_5_a1)

    return input, cnn_5_gap

# 5 CNN layer with residual connections and no activations model
def model_base_5_wide_CNN_noact(input_shape):
    input = Input(shape=input_shape)

    cnn_1_c1 = Conv2D(64, (7, 7), strides=(3, 3), padding='same')(input)
    cnn_1_a = Activation('relu')(cnn_1_c1)

    cnn_2_c1 = Conv2D(64, (5, 5), strides=(3, 3), padding='same')(cnn_1_a)
    #cnn_2_a1 = Activation('relu')(cnn_2_c1)
    cnn_2_c2 = Conv2D(64, (3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    #cnn_2_a2 = Activation('relu')(cnn_2_c2)
    cnn_2_ap = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(cnn_1_a)
    cnn_2_c = Concatenate()([cnn_2_c1, cnn_2_c2, cnn_2_ap])

    cnn_3_c1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(cnn_2_c)
    #cnn_3_a1 = Activation('relu')(cnn_3_c1)
    cnn_3_c2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(cnn_2_c)
    #cnn_3_a2 = Activation('relu')(cnn_3_c2)
    cnn_3_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_2_c)
    cnn_3_c = Concatenate()([cnn_3_c1, cnn_3_c2, cnn_3_ap])

    cnn_4_c1 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(cnn_3_c)
    #cnn_4_a1 = Activation('relu')(cnn_4_c1)
    cnn_4_c2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(cnn_3_c)
    #cnn_4_a2 = Activation('relu')(cnn_4_c2)
    cnn_4_ap = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_3_c)
    cnn_4_c = Concatenate()([cnn_4_c1, cnn_4_c2, cnn_4_ap])

    cnn_5_c1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(cnn_4_c)
    #cnn_5_a1 = Activation('relu')(cnn_5_c1)
    cnn_5_gap = GlobalAveragePooling2D()(cnn_5_c1)

    return input, cnn_5_gap

# ---
# Model heads

def model_head_hidden_dense(model_input, model_output, outputs, model_settings):

    # Main input (image)
    inputs = [model_input]

    x = model_output

    # Add additional inputs with more data and concatenate
    if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
        kmh_input = Input(shape=(1,), name='kmh_input')
        x = Concatenate()([x, kmh_input])
        inputs.append(kmh_input)

    # Add additional fully-connected layer
    x = Dense(model_settings['hidden_1_units'], activation='relu')(x)

    # And finally output (regression) layer
    predictions = Dense(outputs, activation='linear')(x)

    # Create a model
    model = Model(inputs=inputs, outputs=predictions)

    return model

def model_head_direct(model_input, model_output, outputs, model_settings):

    # Main input (image)
    inputs = [model_input]

    x = model_output

    # Add additional inputs with more data and concatenate
    if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
        kmh_input = Input(shape=(1,), name='kmh_input')
        y = Dense(4, activation='relu')(kmh_input)
        x = Concatenate()([x, y])
        inputs.append(kmh_input)

    # And finally output (regression) layer
    predictions = Dense(outputs, activation='linear')(x)

    # Create a model
    model = Model(inputs=inputs, outputs=predictions)

    return model
