import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model

class utilities(object):
    
    def __init__(self):
        self.binary_number_1 = 255
        self.binary_number_2 = float(self.binary_number_1 / 2)
        self.rgb_mean = np.array([0.4488, 0.4371, 0.4040]) * self.binary_number_1
        
    
    def resolve(self, model, low_resolution_image_batch):
        low_resolution_image_batch = tf.cast(low_resolution_image_batch, tf.float32)
        super_resolve_image_batch = model(low_resolution_image_batch)
        super_resolve_image_batch = tf.clip_by_value(super_resolve_image_batch, 0, self.binary_number_1)
        super_resolve_image_batch = tf.cast(tf.round(super_resolve_image_batch), tf.uint8)
        return super_resolve_image_batch
        
    def evaluate(self, model, dataset):
        psnr_values = []
        for low_resolution_image, high_resolution_image in dataset:
            super_resolve_image = self.resolve(model, low_resolution_image)
            psnr_value = self.peak_signal_to_noise_ratio(high_resolution_image, super_resolve_image)[0]
            psnr_values.append(psnr_value)
        return tf.reduce_mean(psnr_values)
        
    def single_resolve(self, model, low_resolution_image):
        return self.resolve(model, tf.expand_dims(low_resolution_image, axis=0))[0]
    
    
    # standard normalization of rbg image 
    def normalize_1(self, image):
        return (image - self.rgb_mean) / self.binary_number_2
    
    # standard denormalization of rbg image 
    def denormalize_1(self, image):
        return (image * self.binary_number_2) + self.rgb_mean
    
    # [0,1] normalization of rbg image 
    def normalize_2(self, image):
        return (image / self.binary_number_1)
    
    # [0,1] denormalization of rbg image 
    def denormalize_2(self, image):
        return (image  * self.binary_number_1)
    
    # [-1,1] normalization of rbg image 
    def normalize_3(self, image):
        return (image / self.binary_number_2) - 1
    
    # [-1,1] denormalization of rbg image 
    def denormalize_3(self, image):
        return (image + 1) * self.binary_number_2
    
    # PSNR
    def peak_signal_to_noise_ratio(self, image_1, image_2):
        return tf.image.psnr(image_1, image_2, max_val=self.binary_number_1)
        
    def pixel_shuffle(self,scaler):
        return lambda x: tf.nn.depth_to_space(image, scaler)
    
    
    
    
# enhanced deep super resolution
class model_creation_edsr(object):
    def __init__(self, scaler, number_filters = 64, number_res_blocks = 8, res_block_scalling = None):
        
        self.scaler = scaler
        self.number_filters = number_filters
        self.number_res_blocks = number_res_blocks
        self.res_block_scalling = res_block_scalling
        
        
    def enhanced_deep_super_resolution(self):
        
        x_input = Input(shape =(None,None,3))
        x = Lambda(utilities.normalize_1)(x_input)
        
        x = temp = Conv2D(self.number_filters, 3, padding='same')(x)
        
        for i in range(self.number_res_blocks):
            temp = self.residual_block(temp)
        
        temp = Conv2D(self.number_filters, 3, padding='same')(temp)
        x = Add()([x,temp])
        
        x = self.upsampling(x)
        x = Conv2D(3, 3, padding='same')(x)
        
        x = Lambda(utilities.denormalize_1)(x)
        
        return Model(x_input, x, name='enhanced_deep_super_resolution')
            
            
    def single_upsampling(self, x, factor, **kwargs):
        x = Conv2D(self.number_filters * (factor ** 2), 3 , padding='same', **kwargs)(x)
        return Lambda(utilities.pixel_shuffle(factor))(x)
    
    
    def upsampling(self, x):
        if self.scaler == 2:
            x = self.single_upsampling(x, 2, name='conv2d_scale_1')
        elif self.scaler == 3:
            x = self.single_upsampling(x, 3, name='conv2d_scale_1')
        elif self.scaler == 4:
            x = self.single_upsampling(x, 2, name='conv2d_scale_1')
            x = self.single_upsampling(x, 2, name='conv2d_scale_2')
            
        return x
                   
                   
    def residual_block(self, x_input):
        x = Conv2D(self.number_filters, 3, padding='same', activation='relu')(x_input)
        x = Conv2D(self.number_filters, 3, padding='same')(x)
        
        if self.res_block_scalling != None:
            x = Lambda(lambda t: t * self.res_block_scalling)(x)
            
        x = Add()([x_input,x])
        return x
   
   

# super resolution GAN
class model_creation_srgan(object):
    def __init__(self, number_filters = 64, number_res_blocks = 16, momentum = 0.8, sliders=1, batchnorm=False):
        
        self.number_filters = number_filters
        self.number_filters_reset = self.number_filters
        self.number_res_blocks = number_res_blocks
        self.momentum = momentum
        self.sliders = sliders
        self.batchnorm = batchnorm
        self.low_resolution_size = 24
        self.high_resolution_size = 96
        self.alpha = 0.2
        
    def upsampling(self, x):
        x = Conv2D((self.number_filters * 4), kernel_size = 3, padding='same')(x)
        x = Lambda(utilities.pixel_shuffle(2))(x)
        return PReLU(shared_axes=[1, 2])(x)
    
    
    def residual_block(self, x_input):
        x = Conv2D(self.number_filters, kernel_size=3, padding='same')(x_input)
        x = BatchNormalization(momentum = self.momentum)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(self.number_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=self.momentum)(x)
        x = Add()([x_input, x])
        return x
    
    
    def discriminator_block(self, x_input):
        x = Conv2D(self.number_filters, kernel_size=3, sliders=self.sliders, padding='same')(x_input)
        if self.batchnorm == True:
            x = BatchNormalization(momentum=self.momentum)(x)
        return LeakyReLU(alpha=self.alpha)(x)
    
    # super resolve residual network
    def generator(self):
        x_input = Input(shape(None,None,3))
        x = Lambda(utilities.normalize_2)(x_input)
        
        x = Conv2D(self.number_filters, kernel_size=9, padding='same')(x)
        x = temp = PReLU(shared_axes=[1, 2])(x)
        
        for i in range(self.number_res_blocks):
            temp =  self.residual_block(temp)
        
        temp = Conv2D(self.number_filters, kernel_size=3, padding='same')(temp)
        temp = BatchNormalization()(temp)
        x = Add()([x, temp])
        
        x = self.upsampling(x)
        x = self.upsampling(x)
        
        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        # try discriminator 3 if 2 does not work
        x = Lambda(utilities.denormalize_2)(x)
        
        return Model(x_input,x)
    
    def update(self, multiplier):
        self.number_filters = self.number_filters_reset
        self.number_filters * multiplier
        self.sliders = 2
        self.batchnorm = True
        
    def discriminator(self):
        x_input = Input(shape=(self.high_resolution_size, self.high_resolution_size, 3))
        x = Lambda(utilities.normalize_3)(x_input)
        
        x = discriminator_block(x)
        self.update(1)
        x = discriminator_block(x)
        
        self.update(2)
        x = discriminator_block(x)
        x = discriminator_block(x)
        
        self.update(4)
        x = discriminator_block(x)
        x = discriminator_block(x)
        
        self.update(8)
        x = discriminator_block(x)
        x = discriminator_block(x)
        
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = Dense(1, activation='sigmoid')(x)
        
        return Model(x_input, x)
        


# wide activation for efficient and accurate image super resolution
class model_creation_wdsr(object):
    def __int__(self, scaler, number_filters = 64, number_res_blocks = 8, res_block_expansion = 4, res_block_scaling = None):
        
        self.scaler = scaler
        self.number_filters = number_filters
        self.number_res_blocks = number_res_blocks
        self.res_block_expansion = res_block_expansion
        self.res_block_scaling = res_block_scaling
        self.linear = 0.8
    
        
    def residual_block(self, x_input, block_type):
        
        if block_type == "type 1":
            
            self.res_block_expansion = 4
            x = tfa.layers.WeightNormalization(Conv2D((self.number_filters * self.res_block_expansion), kernel_size=3, padding='same', activation='relu'), data_init=False)(x_input)
            x = tfa.layers.WeightNormalization(Conv2D(self.number_filters, kernel_size=3, padding='same', activation=None), data_init=False)(x)
            
            if self.scaler == True:
                x = Lambda(lambda t: t * self.res_block_scaling)(x)
                
            x = Add()([x_input, x])
            return x
            
            
        elif block_type == "type 2":
            
            self.res_block_expansion = 6
            x = tfa.layers.WeightNormalization(Conv2D((self.number_filters * self.res_block_expansion), kernel_size=1, padding='same', activation='relu'), data_init=False)(x_input)
            x = tfa.layers.WeightNormalization(Conv2D(int(self.number_filters * self.linear), kernel_size=1, padding='same', activation=None), data_init=False)(x)
            x = tfa.layers.WeightNormalization(Conv2D(self.number_filters, kernel_size=3, padding='same', activation=None), data_init=False)(x)
             
            if self.scaler == True:
                x = Lambda(lambda t: t * self.res_block_scaling)(x)
                
            x = Add()([x_input, x])
            return x
            
            
        
    def wide_activation_super_resolution(self):
        x_input = Input(shape(None,None,3))
        x = Lambda(utilities.normalize_2)(x_input)
        
        x_1 = tfa.layers.WeightNormalization(Conv2D(self.number_filters, kernel_size=3, padding='same', activation=None), data_init=False)(x)
        
        for  i in range(self.number_res_blocks):
            x_1 = residual_block(x_1, block_type = 'type 1')
        
        x_1 = tfa.layers.WeightNormalization(Conv2D((3 * (self.scaler**2)), kernel_size=3, padding='same', activation=None), data_init=False)(x_1)
        x_1 = Lambda(utilities.pixel_shuffle(self.scaler))(x_1)
        
        x_2 = tfa.layers.WeightNormalization(Conv2D((3 * (self.scaler**2)), kernel_size=5, padding='same', activation=None), data_init=False)(x_1)
        x_2 = Lambda(utilities.pixel_shuffle(self.scaler))(x_2)
        
        x = Add()([x_1,x_2])
        x = Lambda(utilities.denormalize_1)(x)
        
        return Model(x_input, x)
            
