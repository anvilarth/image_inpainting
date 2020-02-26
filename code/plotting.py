import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def generate_and_save_images(model, test_input, epoch = 33330, saving = False):
    """
    Generate and save image

    In future version this function will be removed!

    Args:
        model (tf.keras.Model): Model, which creating images
        test_input (tf.Tensor): Input for chosen models. Must be 4D-shape
        epoch (int): Number of epoch. Necessary in process of saving
        saving (boolean): If set to True we save every image otherwise no saving

    Return:
        Showing result
    """
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(12,8))

    plt.imshow(tf.cast(tf.squeeze(predictions) * 127.5 + 127.5, tf.uint8))
    plt.axis('off')
    if saving:
        plt.savefig('../image_saves3/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def plot_all_images(model, image):
    """
    Plot all images, which are neccessary for visual analysis

    Args:
        model(tf.keras.Model): Model, which creating images
        image (tf.Tensor): Result of applying read_labels (look preprocess.py)

    Return:
        5 images: Mask, Edge map, Image with mask, Restored image, Real image
    """
    inputs = image[:, :, :5]
    true = image[:, :, 5:]
    edge_map = image[:, :, 3]
    mask = image[:, :, 4]
    input_image = image[:, :, :3]

    inputs_exp = tf.reshape(inputs, (1, 224, 224, 5))
    pred = model.predict(inputs_exp)
    pred_reshape = pred.reshape(224,224, 3)

    fig, ax = plt.subplots(1, 5, figsize = (30, 6))
    ax[0].imshow(mask, aspect = 'auto')
    ax[0].axis('off')

    ax[1].imshow(edge_map, aspect = 'auto')
    ax[1].axis('off')

    ax[2].imshow(tf.cast(input_image * 127.5 + 127.5, tf.uint8), aspect = 'auto')
    ax[2].axis('off')

    ax[3].imshow((pred_reshape * 127.5 + 127.5).astype(np.uint8), aspect = 'auto')
    ax[3].axis('off')

    ax[4].imshow(tf.cast(true * 127.5 + 127.5, tf.uint8), aspect = 'auto')
    ax[4].axis('off')

    plt.show()
