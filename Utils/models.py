from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, LeakyReLU, Input, Activation, concatenate
from tensorflow.keras.applications import MobileNetV2
from keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def model_main_with_mobilenet(image_shape=(224, 224, 3)):
    weight_init = RandomNormal(stddev=0.02)
    net_input = Input((image_shape))
    mobile_net_base = MobileNetV2(
        include_top=False,
        input_shape=image_shape,
        weights='imagenet'
    )
    mobilenet = mobile_net_base(net_input)
    
    conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(net_input)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    
    conv2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv2)
    conv3 =  Activation('relu')(conv3)

    conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv3)
    conv4 = Activation('relu')(conv4)

    conv4_ = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(conv4)
    conv4_ = Activation('relu')(conv4_)

    conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv4_)
    conv5 = Activation('relu')(conv5)

    conv5_ = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv5)
    conv5_ = Activation('relu')(conv5_)
    
    conc = concatenate([mobilenet, conv5_])
    fusion = Conv2D(512, (1, 1), padding='same', kernel_initializer=weight_init)(conc)
    fusion = Activation('relu')(fusion)
    
    skip_fusion = concatenate([fusion, conv5_])
    
    decoder = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_fusion)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    skip_4_drop = Dropout(0.25)(conv5)
    skip_4 = concatenate([decoder, skip_4_drop])
    
    decoder = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_4)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    skip_3_drop = Dropout(0.25)(conv4_)
    skip_3 = concatenate([decoder, skip_3_drop])
    
    decoder = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_3)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    decoder = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    decoder = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(decoder)
    decoder = Activation('relu')(decoder)
    
    output_layer = Conv2D(2, (1, 1), activation='tanh')(decoder)

    model = Model(net_input, output_layer)
    model.compile(Adam(lr=0.0003), loss='mse', metrics=['accuracy'])
    
    return model
