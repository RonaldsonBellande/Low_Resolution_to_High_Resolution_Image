import tensorflow as tf
import tensorflow.compat.v1 as tf2
import numpy as np
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Single Image Super-Resulution Base
class RCNN(object):
    def __init__(self,learning_rate = 1e-3,verbose = True):
        
        self.learning_rate = learning_rate
        self.input_size_1 = 240
        self.input_size_2 = 320
        self.label_size_1 = 240
        self.label_size_2 = 320
        self.verbose = verbose
        self.parameters = []
        self.create_placeholders()
        self.build_model()
        self.create_loss_function()
        self.create_optim()
        self.print_vars()
        #self.print_accuracy_loss_function()
        
        
    def conv2d(self,x,W,padding='VALID',strides=[1,1,1,1]):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID',name='conv')
    
    def print_vars(self):
        for p in self.parameters:
            print(p.name,p.get_shape())
            
    def bias_variable(self,shape):
        bias =tf2.get_variable(dtype=tf.float32,initializer=tf.constant_initializer(0.1),shape=shape,name='bias')
        return bias
    
    def weight_variable(self,shape):
        
        W = tf2.get_variable(dtype=tf.float32,initializer=tf2.truncated_normal_initializer(mean=0.0,stddev=0.001),shape=shape,name='filter')
        return W
    
    def create_placeholders(self):
        self.X = tf2.placeholder(dtype=tf.float32,shape=[None,self.input_size_1,self.input_size_2,3])
        self.y = tf2.placeholder(dtype=tf.float32,shape=[None,self.label_size_1,self.label_size_2,3])
        if self.verbose:
            print("Created placeholders for input and labels")
            
    def build_model(self):
        
        with tf2.variable_scope('conv1'):
            W_conv1 = self.weight_variable([9,9,3,64])
            b_conv1 = self.bias_variable([64])
            f_conv1 = tf.nn.relu(self.conv2d(self.X,W_conv1) + b_conv1)
            self.parameters += [W_conv1,b_conv1]
            
        with tf2.variable_scope('conv2'):
            W_conv2 = self.weight_variable([1,1,64,32])
            b_conv2 = self.bias_variable([32])
            f_conv2 = tf.nn.relu(self.conv2d(f_conv1,W_conv2) + b_conv2)
            self.parameters += [W_conv2,b_conv2]
            
        with tf2.variable_scope('conv3'):
            W_conv3 = self.weight_variable([5,5,32,3])
            b_conv3 = self.bias_variable([3])
            self.f_out = self.conv2d(f_conv2,W_conv3) + b_conv3
            self.parameters += [W_conv3,b_conv3]
            
            
    def create_loss_function(self):
        self.loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.f_out,self.y))))
        if self.verbose:
            print("Loss op created")
            
    def create_optim(self):
        self.optim = tf.optimizers.Adam(learning_rate=self.learning_rate)
        if self.verbose:
            print("Optimizer op created")
            
if __name__=="__main__":
    src = RCNN()
