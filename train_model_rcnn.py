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
from os.path import basename


class Mode_Training_RCNN(object):
    def __init__(self, input_path, save_file_path, nImgs, nEpochs, batch_size, print_every):
        
        self.input_path = input_path
        self.save_file_path = save_file_path
        self.nImgs = nImgs
        self.nEpochs = nEpochs 
        self.batch_size = batch_size
        self.image_naming = None
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
        
    
    def save_image(self, img, file_name, image_to_save):

        if image_to_save == "image":
            image_output = "higher_resolution_images_data_1/image"
        else:
            image_output = "higher_resolution_images_data_1/label"
        
        for i in range(len(self.data_files)):
            cv2.imwrite(os.path.join(image_output, str(file_name)+ ".jpg"), img)
            cv2.waitKey(0)
        
    def read_batch(self):
        imgs = []
        labels = []
        idx = np.random.choice(self.nImgs,self.batch_size)
        for i in idx:
            imgs.append(cv2.imread(self.data_files[i]))
            labels.append(cv2.imread(self.label_files[i]))
            #print(idx)
        imgs,labels = np.array(imgs),np.array(labels)
        imgs = (imgs - self.mean)/self.stddev
        labels = (labels - self.mean)/self.stddev
        now = datetime.datetime.now()
        name = now.strftime("%Y-%m-%d_%H:%M")
        self.save_image(imgs, name, image_to_save = "image")
        return imgs,labels
    
    
    def train(self):
        sess = tf.compat.v1.Session()
        sess.run(tf2.initialize_all_variables())
        for i in range(self.nIters):
            imgs_inp,imgs_lab = self.read_batch()
            print(self.nIters)
            #self.save_image(imgs_inp, "temp", image_to_save = "image")
            #_,curr_loss = sess.run([self.model.optim,self.model.loss],feed_dict={self.model.X:imgs_inp,self.model.y:imgs_lab})
            #if(i%self.print_every==0):
                #print("Step {0} Training loss: {1}".format(i+1,curr_loss))
        #save = raw_input("Save the model?")
        if('y'=="y"):
            now = datetime.datetime.now()
            name = now.strftime("%Y-%m-%d_%H:%M")
            saver = tf2.train.Saver()
            path = saver.save(sess,self.save_file_path + '/' + name)
            print("Saved in " + path)


if __name__=="__main__":
    train_model = Mode_Training_RCNN("images_data_2/","higher_resolution_images_data_1/logs/",50,10,16,10)
    train_model.train()
