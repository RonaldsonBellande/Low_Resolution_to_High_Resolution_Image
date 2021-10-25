from header_imports import *
from computer_vision_model_building import *

class computer_vision_training(object):
    def __init__(self, model_type, image_type, category):
        
        self.number_classes = 43
        self.model_type = str(model_type)
        self.image_type = str(image_type)
        self.category = str(category)
        
        # Category 1
        computer_vision_building_obj = computer_vision_building(model_type = self.model_type, image_type = self.image_type, category = self.category)
        self.model = computer_vision_building_obj.get_model()
        
        xy_data = computer_vision_building_obj.get_data()

        self.X_train = xy_data[0]
        self.Y_train = xy_data[1]
        self.X_test = xy_data[2]
        self.Y_test = xy_data[3]
        self.Y_test_vec = xy_data[4]
        
        self.batch_size = [10, 20, 40, 60, 80, 100]
        self.epochs = [1, 5, 15, 50, 100, 200]
        self.param_grid = dict(batch_size = self.batch_size, epochs = self.epochs)
        self.callbacks = keras.callbacks.EarlyStopping(monitor='val_acc', patience=4, verbose=1)
        
        self.earlyStop = EarlyStopping(patience=2)
        self.learining_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor= 0.5,min_lr=0.00001)
        
        self.callbacks_2 = self.earlyStop, self.learining_rate_reduction
        
        # Model
        self.model_categories = computer_vision_building_obj.get_categories()
        
        # Train
        self.name = "category_1"
        self.train_model()
        self.evaluate_model()
        self.plot_model()
        self.plot_random_examples()
        


    #  Training model 
    def train_model(self):
       
        grid = GridSearchCV(estimator = self.model, param_grid = self.param_grid, n_jobs = 1, cv = 3, verbose = 10)
        
        # Determine where the training time starts
        start = "starting --: "
        self.get_training_time(start)

        self.computer_vision_model = self.model.fit(self.X_train, self.Y_train,
                batch_size=self.batch_size[2],
                validation_split=0.10,
                epochs=self.epochs[3],
                callbacks=[self.callbacks_2],
                shuffle=True)

        # Determine when the training time ends
        start = "ending --: " 
        self.get_training_time(start)
        
        self.model.save_weights("models/" + self.image_type + "_" + self.category + "_" + self.name + "_" + self.model_type + "_computer_vision_categories_"+ str(self.number_classes)+"_model.h5")
   

    # Evaluate model
    def evaluate_model(self):
        evaluation = self.model.evaluate(self.X_test, self.Y_test, verbose=1)

        with open("graph_charts/" + self.image_type + "_" + self.name + "_" + self.category + "_" + self.model_type + "_evaluate_computer_vision_category_" + str(self.number_classes) + ".txt", 'w') as write:
            write.writelines("Loss: " + str(evaluation[0]) + "\n")
            write.writelines("Accuracy: " + str(evaluation[1]))
        
        print("Loss:", evaluation[0])
        print("Accuracy: ", evaluation[1])



    # PLotting model
    def plot_model(self):

        # Computer Vision modeling
        plt.plot(self.computer_vision_model.history['accuracy'])
        plt.plot(self.computer_vision_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig("graph_charts/" + self.image_type + "_" + self.category + "_" + self.name + "_" + self.model_type + '_accuracy_' + str(self.number_classes) + '.png', dpi =500)


        plt.plot(self.computer_vision_model.history['loss'])
        plt.plot(self.computer_vision_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig("graph_charts/" + self.image_type + "_" + self.category + "_" + self.name + "_" + self.model_type + '_lost_' + str(self.number_classes) +'.png', dpi =500)



    def plot_random_examples(self):

        plt.figure( dpi=256)
        predicted_classes = self.model.predict_classes(self.X_test)

        for i in range(100):
            plt.subplot(10,10,i+1)
            fig=plt.imshow(self.X_test[i,:,:,:])
            plt.axis('off')
            plt.title("Predicted - {}".format(self.model_categories[predicted_classes[i]] ) + "\n Actual - {}".format(self.model_categories[self.Y_test_vec[i,0]] ),fontsize=1)
            plt.tight_layout()
            plt.savefig("graph_charts/" + self.image_type + "_" + self.category + "_" + self.name + "_" + self.model_type + '_prediction' + str(self.number_classes) + '.png', dpi =500)



    # Record time for the training
    def get_training_time(self, start):

        date_and_time = datetime.datetime.now()
        test_date_and_time = "/test_on_date_" + str(date_and_time.month) + "_" + str(date_and_time.day) + "_" + str(date_and_time.year) + "_time_at_" + date_and_time.strftime("%H:%M:%S")

        with open("graph_charts/" + self.image_type + "_" + self.category + "_" + self.name + "_" + self.model_type + "_evaluate_training_time_" + str(self.number_classes) + ".txt", 'a') as write:
            write.writelines(start + test_date_and_time + "\n")




