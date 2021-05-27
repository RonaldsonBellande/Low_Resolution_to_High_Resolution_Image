import tensorflow as tf
import numpy as np
import cv2
from image_conversion_rcnn import *
import os
from os.path import basename
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from glob import glob
tf.compat.v1.disable_eager_execution()
import datetime


class Mode_Training_RCNN(object):
    def __init__(self, input_path, save_file_path, nImgs, nEpochs, batch_size, print_every):
        
        self.input_path = input_path
        self.save_file_path = save_file_path
        self.nImgs = nImgs
        self.nEpochs = nEpochs 
        self.batch_size = batch_size
        self.print_every = print_every
        self.nIters = int(self.nImgs*self.nEpochs/self.batch_size)
        self.data_files = [count for count in glob(self.input_path +'*') if 'jpg' in count]
        self.label_files = [count for count in glob(self.input_path +'*') if 'jpg' in count]
        print("{0} files loaded".format(len(self.data_files)))
        self.mean = 113.087154872
        self.stddev = 69.7176496121
        self.model = RCNN()
        self.imgs = []
        self.target = []
        
        
    def read_batch(self):
        imgs = []
        labels = []
        idx = np.random.choice(self.nImgs,self.batch_size)
        for i in idx:
            imgs.append(cv2.imread(self.data_files[i]))
            labels.append(cv2.imread(self.label_files[i]))
        imgs,labels = np.array(imgs),np.array(labels)
        imgs = (imgs - self.mean)/self.stddev
        labels = (labels - self.mean)/self.stddev
        return imgs,labels
    
    
    def train(self):
        sess = tf.compat.v1.Session()
        sess.run(tf2.initialize_all_variables())
        for i in range(self.nIters):
            imgs_inp,imgs_lab = self.read_batch()
            _,curr_loss = sess.run([self.model.optim,self.model.loss],feed_dict={self.model.X:imgs_inp,self.model.y:imgs_lab})
            if(i%self.print_every==0):
                print("Step {0} Training loss: {1}".format(i+1,curr_loss))
        save = raw_input("Save the model?")
        if(save=="y"):
            now = datetime.datetime.now()
            name = now.strftime("%Y-%m-%d_%H:%M")
            saver = tf.train.Saver()
            path = saver.save(sess,os.environ['HOME'] + '/' + name + '.ckpt')
            print("Saved in " + path)


if __name__=="__main__":
    train_model = Mode_Training_RCNN("images_data_2/","higher_resolution_images_data_1/",50,10,16,10)
    train_model.train()
