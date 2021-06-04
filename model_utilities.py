import numpy as np
import tensorflow as tf
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
        for low_resolution_image, hr_image in dataset:
            super_resolve_image = self.resolve(model, low_resolution_image)
            psnr_value = self.peak_signal_to_noise_ratio(hr_image, super_resolve_image)[0]
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
    
    
    
        
class model_creation(object):
    def __init__(self, scaler, number_filters = 64, number_res_blocks = 8, res_block_scalling = None):
        
        self.scaler = scaler
        self.number_filters = number_filters
        self.number_res_blocks = number_res_blocks
        self.res_block_scalling = res_block_scalling
        
        
    def enhanced_deep_super_resolution(self):
        
        x_in = Input(shape =(None,None,3))
        x = Lambda(utilities.normalize_1)(x_in)
        
        x = b = Conv2D(self.number_filters, 3, padding='same')(x)
        
        for i in range(self.number_res_blocks):
            b = self.res_block(b)
        
        b = Conv2D(self.number_filters, 3, padding='same')(b)
        x = Add()([x,b])
        
        x = self.upsampling(x)
        x = Conv2D(3, 3, padding='same')(x)
        
        x = Lambda(utilities.denormalize_1)(x)
        
        return Model(x_in, x, name='enhanced_deep_super_resolution')
            
            
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
                   
                   
    def res_block(self, x_in):
        x = Conv2D(self.number_filters, 3, padding='same', activation='relu')(x_in)
        x = Conv2D(self.number_filters, 3, padding='same')(x)
        
        if self.res_block_scalling != None:
            x = Lambda(lambda t: t * self.res_block_scalling)(x)
            
        x = Add()([x_in,x])
        return x
        
        
