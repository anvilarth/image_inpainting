import tensorflow as tf
import time

discr_loss = tf.keras.metrics.Mean(name='dicriminator_loss')
gene_loss = tf.keras.metrics.Mean(name='generator_loss')

@tf.function()
def train_step(real_image, fake_image, update_d):
    """

    Applying optimization step for GAN model

    Args:
        real_image: Image from real dataset
        fake_image: Generated image
        update_d: Rule for updating discriminator. If update_d set to True we update discriminator otherwise no update.

    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(fake_image, training = True)

        real_output = discriminator(real_image, training = True)
        fake_output = discriminator(generated_image, training = True)

        outputs_real = extractor(real_image)
        outputs_fake = extractor(generated_image)

        gen_loss = generator_loss(real_image,  generated_image, fake_output, outputs_real, outputs_fake)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    if update_d:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    if update_d:
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    gene_loss(gen_loss)
    discr_loss(disc_loss)

def train(dataset, epochs):
    """

    Starting training loop

    Args:
        dataset (tf.Dataset): Dataset for training
        epochs (int): Number of training epochs

    """
    update_d = True
    for epoch in range(epochs):
        start = time.time()
        for (in_image, true_image) in dataset:
            train_step(true_image, in_image, update_d)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print ('Generator loss {}, Discriminator loss {} '.format(gene_loss.result(), discr_loss.result()))

        generate_and_save_images(generator, setok, epoch)

        generator.save('../generator_saves3/generator--{:02d}-{:.4f}.h5'.format(epoch, gene_loss.result()))
        discriminator.save('../discriminator_saves3/discriminator--{:02d}-{:.4f}.h5'.format(epoch, discr_loss.result()))

        if discr_loss.result() <= 0.3 * 0.999 **(epoch):
            update_d = False
        else: update_d = True

        discr_loss.reset_states()
        gene_loss.reset_states()
