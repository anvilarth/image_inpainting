import tensorflow as tf
from models import *
from loss import *
from plotting import *
from train import *
from preprocess import *

image_size = 224
lr_d = 0.0001
lr_g = 0.0001
batch_size = 16
beta1 = 0.5

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block4_conv2', 'block5_conv2']

path_ds, all_image_paths = get_dataset('/home/anvilarth/ML/Dataset/arcDataset')

buffer_size = len(all_image_paths)

image_ds = path_ds.map(preprocess_image)
ds = image_ds.shuffle(buffer_size).batch(batch_size).prefetch(-1)

discriminator = make_discriminator_model(image_size)
generator = make_generator_model(image_size)


extractor = StyleContentModel(style_layers, content_layers)


discriminator_optimizer = tf.keras.optimizers.Adam(lr_d, beta_1 = beta1)
generator_optimizer = tf.keras.optimizers.Adam(lr_g, beta_1= beta1)


check = read_labels(all_image_paths[100])
innos = check[:, :, :5]
setok = tf.reshape(innos, (1, image_size, image_size, 5))

image = read_labels(all_image_paths[100])

plot_all_images(generator, image)

generator = load_model('Saved_models/generator--89-53.9974.h5')

plot_all_images(generator, image)
