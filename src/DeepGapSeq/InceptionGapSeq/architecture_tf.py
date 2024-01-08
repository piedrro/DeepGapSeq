from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

def inception_module(input_tensor, 
                    n_filters=40, 
                    bottleneck_channels=32, 
                    kernel_sizes=[9, 19, 39], 
                    activation='relu', 
                    stride=1):
   
    bottleneck = Conv1D(filters=bottleneck_channels, kernel_size=1, 
                        padding='same', activation=activation,use_bias=False)(input_tensor)
    
    conv_branches = []
    for i in range(len(kernel_sizes)):
            conv_branches.append(Conv1D(filters=n_filters, kernel_size=kernel_sizes[i],
                                        strides=stride, padding='same', activation=activation, use_bias=False)(bottleneck))
    
    max_pool = MaxPooling1D(pool_size=3, strides=stride, padding='same')(input_tensor)
    conv_1x1_branch = Conv1D(filters=n_filters, kernel_size=1, padding='same', activation=activation, use_bias=False)(max_pool)
    
    conv_branches.append(conv_1x1_branch)

    x = Concatenate(axis=2)(conv_branches)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                        padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

def build_model(input_channel,
                n_classes,
                use_shortcut=True,
                depth=6):
    input_layer = Input(shape=(None,input_channel))
    x = input_layer
    input_res = input_layer
    
    for d in range(depth):
        x = inception_module(x)
        if use_shortcut and d % 3 == 0 and d > 0:
            x = _shortcut_layer(input_layer,x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)
    output_layer = Dense(n_classes, activation='softmax')(gap_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
    
    

