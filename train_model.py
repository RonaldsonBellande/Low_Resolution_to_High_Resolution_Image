import time
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from model_architecture import *


class Model_Training:
    def __init__(self, model, loss, learning_rate, checkpoint_dir='./checkpoint/edsr', generator=None, discriminator=None, content_loss='VGG54', model_type = 'None'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), psnr=tf.Variable(-1.0), optimizer=Adam(learning_rate), model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint, directory=checkpoint_dir, max_to_keep=3)
        self.restore()
        
        self.model_type = model_type
        
        if self.model_type == 'srgan':
            
            if content_loss == 'VGG22':
                self.vgg = srgan.vgg_22()
            elif content_loss == 'VGG54':
                self.vgg = srgan.vgg_54()
            else:
                raise ValueError("content_loss must be either 'VGG22' or 'VGG54'")

            self.content_loss = content_loss
            self.generator = generator
            self.discriminator = discriminator
            self.generator_optimizer = Adam(learning_rate=learning_rate)
            self.discriminator_optimizer = Adam(learning_rate=learning_rate)
            self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
            self.mean_squared_error = MeanSquaredError()
            
            

    @property
    def model(self):
        return self.checkpoint.model


    def train(self, train_dataset, valid_dataset = None, steps=300000, evaluate_every=1000, save_best_only=False):
        if self.model_type == 'srgan':
            
            pls_metric = Mean()
            dls_metric = Mean()
            step = 0

            for lr, hr in train_dataset.take(steps):
                step += 1

                pl, dl = self.train_step(lr, hr)
                pls_metric(pl)
                dls_metric(dl)

                if step % 50 == 0:
                    print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                    pls_metric.reset_states()
                    dls_metric.reset_states()
            
        else:
            
            loss_mean = Mean()
            checkpoint_mgr = self.checkpoint_manager
            checkpoint = self.checkpoint

            self.now = time.perf_counter()

            for lr, hr in train_dataset.take(steps - checkpoint.step.numpy()):
                checkpoint.step.assign_add(1)
                step = checkpoint.step.numpy()

                loss = self.train_step(lr, hr)
                loss_mean(loss)

                if step % evaluate_every == 0:
                    loss_value = loss_mean.result()
                    loss_mean.reset_states()

                    # Compute PSNR on validation dataset
                    psnr_value = self.evaluate(valid_dataset)

                    duration = time.perf_counter() - self.now
                    print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                    if save_best_only and psnr_value <= checkpoint.psnr:
                        self.now = time.perf_counter()
                        # skip saving checkpoint, no PSNR improvement
                        continue

                    checkpoint.psnr = psnr_value
                    checkpoint_mgr.save()

                    self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        
        if self.model_type == 'srgan':
            
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                lr = tf.cast(lr, tf.float32)
                hr = tf.cast(hr, tf.float32)

                sr = self.generator(lr, training=True)

                hr_output = self.discriminator(hr, training=True)
                sr_output = self.discriminator(sr, training=True)

                con_loss = self._content_loss(hr, sr)
                gen_loss = self._generator_loss(sr_output)
                perc_loss = con_loss + 0.001 * gen_loss
                disc_loss = self._discriminator_loss(hr_output, sr_output)

            gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            return perc_loss, disc_loss
            
        else:
            with tf.GradientTape() as tape:
                lr = tf.cast(lr, tf.float32)
                hr = tf.cast(hr, tf.float32)

                sr = self.checkpoint.model(lr, training=True)
                loss_value = self.loss(hr, sr)

            gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
            self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

            return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
            
            
    @tf.function            
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss


class Enhanced_DeepSuper_Resolution_Model_Training(Model_Training):
    def __init__(self,model, checkpoint_dir,learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, model_type='edsr')

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class Wide_Activation_Super_Resolution_Model_Training(Model_Training):
    def __init__(self, model, checkpoint_dir, learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, model_type='wdsr')

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class Super_Resolution_GAN_Model_Training(Model_Training):
    def __init__(self, model, checkpoint_dir, learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, model_type='srgan')

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)
