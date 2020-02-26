from tensorflow.keras.layers import LeakyReLU, ReLU, Activation, GaussianNoise, Flatten, Add
from tensorflow.keras.layers import Lambda, Dense, Input, Conv2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model, load_model


def decode_block(im_in, filter_size):
    """

    Decode block is simplification for writing U-net (Decoding part)

    Args:
        im_in (tf.Tensor): Input image for decode_block
        filter_size (int): Amount of filters using for convolutions

    Returns: tf.Tensor, tf.Tensor
        Result of convolutions and intermediate layer for skip-connection

    """
    conv1 = Conv2D(filter_size, 3, padding = 'same', kernel_initializer = 'he_normal')(im_in)
    relu11 = LeakyReLU(0.2)(conv1)
    conv2 = Conv2D(filter_size, 3, padding = 'same', kernel_initializer = 'he_normal')(relu11)
    sum1 = Add()([conv2, conv1])
    relu12 = LeakyReLU(0.2)(sum1)
    down1 = Conv2D(filter_size * 2, 3, strides = 2,  padding = 'same', kernel_initializer = 'he_normal')(relu12)
    relu13 = LeakyReLU(0.2)(down1)

    return relu13, relu12


def encode_block(im_in, res_in, filter_size):
    """

    Encode block is simplification for writing U-net (Encoding part)

    Args:
        im_in (tf.Tensor): Input image for encode_block
        res_in (tf.Tensor): Intermediate layer from skip-connection
        filter_size (int): Amount of filters using in process

    Returns: tf.Tensor
        Result of applying convolutions
    """
    conv1 = Conv2D(filter_size, 3, padding = 'same', kernel_initializer = 'he_normal')(im_in)
    relu11 = LeakyReLU(0.2)(conv1)
    conv2 = Conv2D(filter_size, 3, padding = 'same', kernel_initializer = 'he_normal')(relu11)
    sum1 = Add()([conv2, conv1])
    relu12 = LeakyReLU(0.2)(sum1)
    upsampl1 = UpSampling2D(2)(relu12)
    up1 = Conv2D(filter_size // 2, 3, padding = 'same', kernel_initializer = 'he_normal')(upsampl1)
    sum1 = concatenate([up1, res_in])
    relu12 = LeakyReLU(0.2)(sum1)

    return relu12


def make_generator_model(image_size = 224):
    """

    Creating generator as U-net

    Args:
        image_size (int): Size of input image. Recomended to set image_size as default value

    Returns: tf.keras.Model
        Generator model
    """
    inputs = Input((image_size, image_size, 5))

    relu1, conv1 = decode_block(inputs, 8)
    relu2, conv2 = decode_block(relu1, 16)
    relu3, conv3 = decode_block(relu2, 32)
    relu4, conv4 = decode_block(relu3, 64)
    relu5, conv5 = decode_block(relu4, 128)

    relu6 = encode_block(relu5, conv5, 256)
    relu7 = encode_block(relu6, conv4, 128)
    relu8 = encode_block(relu7, conv3, 64)
    relu9 = encode_block(relu8, conv2, 32)
    relu10 = encode_block(relu9, conv1, 16)

    last = Conv2D(3, 3, padding = 'same', kernel_initializer = 'he_normal')(relu10)
    part_input = Lambda(lambda x: x[:, :, :, :3])(inputs)
    sum_last = Add()([last, part_input])
    outputs = Activation('tanh')(sum_last)

    return Model(inputs = inputs, outputs = outputs)


def disc_block(im_in, filter_size):
    """

    Disc block is simplification for writing discriminator

    Args:
        im_in (tf.Tensor): Input image for disc_block
        filter_size (int): Amount of filters in using

    Returns:
        Result of applying (Conv -> LReLU(0.2) - >Conv -> LReLU(0.2))

    """
    conv1 = Conv2D(filter_size, 3, padding = 'same', kernel_initializer = 'he_normal')(im_in)
    relu11 = LeakyReLU(0.2)(conv1)
    down1 = Conv2D(filter_size * 2, 3, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(relu11)
    relu12 = LeakyReLU(0.2)(down1)
    return relu12


def make_discriminator_model(image_size = 224):
    """
    Creating discriminator as simple Conv-net with Downsamplings

    Args:
        image_size (int): Size of input image. Recomended to set image_size as default value

    Returns: tf.keras.Model
        Probability of input image is real
    """
    inputs = Input((image_size, image_size,3))

    noise_input =  GaussianNoise(stddev= 0.02)(inputs)
    relu1 = disc_block(noise_input, 8)
    relu2 = disc_block(relu1, 16)
    relu3 = disc_block(relu2, 32)
    relu4 = disc_block(relu3, 64)
    relu5 = disc_block(relu4, 128)
    relu6 = disc_block(relu5, 256)

    flat = Flatten()(relu6)
    outputs = Dense(1, activation = 'sigmoid')(flat)

    return Model(inputs = inputs, outputs = outputs)
