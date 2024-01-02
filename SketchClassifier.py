import tensorflow as tf
from tensorflow import keras
from keras.layers import MaxPooling2D, Dropout, Dense,Flatten, Conv2D, InputLayer
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import os
import time
import sys


def create_model(param_dict, num_classes):
    model = keras.models.Sequential()
    diff = len(param_dict["conv_layers"])-len(param_dict["pooling_layers"])
    if diff>0:
        for i in range(0,diff):
            param_dict["pooling_layers"].append(None)
   
    model.add(InputLayer(input_shape=(28,28,1)))
    for conv_tuple, pool_tuple in zip(param_dict["conv_layers"],param_dict["pooling_layers"]):
        model.add(Conv2D(conv_tuple[0],conv_tuple[1],activation="relu"))
        if pool_tuple is not None:
            model.add(MaxPooling2D(pool_tuple[0],pool_tuple[1],padding="same"))

    model.add(Flatten())
    
    diff = len(param_dict["dense_layers"])-len(param_dict["dropout_layers"])
    if diff>0:
        for i in range(0,diff):
            param_dict["dropout_layers"].append(None)
    for dense, dropout in zip(param_dict["dense_layers"], param_dict["dropout_layers"]):
        model.add(Dense(dense,activation="relu"))
        if dropout is not None:
            model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def load_data(size=10000,data_folder="datasets"):
  data_files = [file for file in os.listdir(data_folder) if file.endswith(".npy")]
  data_sets = [np.load(os.path.join(data_folder, file)) for file in data_files]
  print(data_files)
  data=[data_set[:size] for data_set in data_sets]
  data=np.concatenate(data,axis=0)
  data=data.reshape((data.shape[0],28,28,1))
  labels=[np.full((size,),i) for i in range(0,len(data_sets))]
  labels=np.concatenate(labels,axis=0)
  class_names = [name.replace("full_numy_bitmap_","",1).replace("_","").replace(".npy","") for name in data_files]

  # normalize the entries of the array to be between 0 to 1
  data = data / 255
  print(data.shape, labels.shape)
  return data,labels, class_names



def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels



def main(param_dict):

    data, labels, class_names = load_data(size=param_dict["data_size"],data_folder='data_sets')
    num_classes = len(class_names)
    if param_dict["shuffle_mode"]==1:
        data, labels = sklearn.utils.shuffle(data, labels)
        shuffle=False
    elif param_dict["shuffle_mode"]==2:
        shuffle=True
    elif param_dict["shuffle_mode"]==0: 
        shuffle=False
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, random_state=0 # makes sure the data is always split in the same way 
                                                        ,test_size=param_dict["training_to_test_ratio"])

    model = create_model(param_dict,num_classes)
    print(training_data.shape, training_labels.shape,testing_data.shape,testing_labels.shape)
    print(model.summary())
    tensorboard = keras.callbacks.TensorBoard(log_dir=f"{param_dict['model_dir']}", histogram_freq=1)
    model.fit(training_data, training_labels, validation_data=(testing_data, testing_labels), epochs=param_dict["total_epochs"], batch_size=param_dict["batch_size"],shuffle=shuffle, callbacks=[tensorboard])
    model.save(f'{param_dict["model_dir"]}\\{param_dict["model_name"]}.h5')
    if param_dict["save_param_dict"]:
        with open(f"{param_dict['model_dir']}\\{param_dict['model_name']}_conf.txt","w") as f:
            f.write(str(param_dict))




def test(model_name):
    from draw_board import SketchApp, Preprocess
    model = keras.models.load_model(model_name)
    # Gets a 28 by 28 numpy array and returns the model's guess
    def predict(image):
        pre= Preprocess(image) 
        prediction_matrix = model.predict(pre)
        prediction_index = np.argmax(prediction_matrix[0])
        prediction = class_names[prediction_index] + f" with {prediction_matrix[0][prediction_index]*100}% certainty"

        return prediction
        
    canvas = SketchApp(predict)
    canvas.run()

param_dict= {
    # Model parameters
    "conv_layers": [(64,(5,5)),(64,(5,5))], # Enter the amount of filters and kernel size as a tuple
    "pooling_layers": [((2,2),(2,2)),((2,2),(2,2))], # Enter the pooling size and stride amount as a tuple. Add None to skip a layer.
    "dense_layers": [512,128], # Enter the amount of neurons in the dense layers
    "dropout_layers": [0.3,0.15], # Leave empty if you do not want dropout layers, enter the dropout rate otherwise. Add None to skip a layer.
      
    # Training Parameters
    "data_size": 50000  , # data size
    "data_augmentation": False , # Whether to augment the data or not
    "training_to_test_ratio": 0.2 , # Training data to testing data ratio. Put 0 to train the model on all the data
    "batch_size": 64 , # Batch size
    "total_epochs": 10 , # Amount of epochs
    "shuffle_mode": 2 , # Put 0 to not shuffle the data, 1 to shuffle the data once at the start, 2 to shuffle during each epoch
    
    # Saving parameters 
    "model_dir": "final_project_present" , # save model dict
    "model_name": "30-15-dropout" , # Name of the model
    "save_param_dict": True # Whether or not to save the param dict as a file
    
    }       
if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "train":
        try:
            train(param_dict)
        except BaseException as e:
            print("Error:\n"+str(e))
            input("Press anything to close this window.")
        input("Training is done. Press anything to close this window.")
    elif arg == "test":
        model_name = sys.argv[2]
        try:
            test(model_name)
        except BaseException as e:
            print("Error:\n"+str(e))
            input("Press anything to close this window.")
         
    
