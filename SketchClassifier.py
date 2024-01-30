import tensorflow as tf
from tensorflow import keras
from keras.layers import MaxPooling2D, Dropout, Dense,Flatten, Conv2D, InputLayer
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import os
import time
from tqdm.auto import tqdm


# Function to create the model
def create_model(param_dict, num_classes):
    model = keras.models.Sequential()

    # Input layer
    model.add(InputLayer(input_shape=(28,28,1)))

    # Pad the pooling_layers list to be the size of the conv_layers list
    diff = len(param_dict["conv_layers"])-len(param_dict["pooling_layers"])
    if diff>0:
        for i in range(0,diff):
            param_dict["pooling_layers"].append(None)

    diff = len(param_dict['conv_layers']) - len(param_dict['conv_dropout_layers'])
    if diff>0:
      for i in range(0,diff):
        param_dict['conv_dropout_layers'].append(None)
    # Add the conv_layers and pool_layers
    for conv_tuple, pool_tuple, dropout_rate in zip(param_dict["conv_layers"],param_dict["pooling_layers"], param_dict['conv_dropout_layers']):
        model.add(Conv2D(conv_tuple[0],conv_tuple[1],activation="relu"))
        if pool_tuple is not None:
            model.add(MaxPooling2D(pool_tuple[0],pool_tuple[1],padding="same"))
        if dropout_rate is not None:
          model.add(Dropout(dropout_rate))


    # Flatten layer
    model.add(Flatten())

    # Pad the dropout_layers list to be the size of the dense_layers list
    diff = len(param_dict["dense_layers"])-len(param_dict["dropout_layers"])
    if diff>0:
        for i in range(0,diff):
            param_dict["dropout_layers"].append(None)

    # Add the dense_layers and dropout_layers
    for dense, dropout in zip(param_dict["dense_layers"], param_dict["dropout_layers"]):
        model.add(Dense(dense,activation="relu"))
        if dropout is not None:
            model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    return model

# Helping function to turn numpy array pairs of data and labels into a single tfrecord file
def numpy_to_tfrecord(data,labels,filename):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(image, label):
        feature = {
            'image': _bytes_feature(image.tobytes()),
            'label': _int64_feature(label),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with tf.io.TFRecordWriter(filename) as writer:
      for i in range(len(data)):
          tf_example = serialize_example(data[i], labels[i])
          writer.write(tf_example)
    
def write_data(size_per_class=10000,split_size=100,training_to_test_ratio=0.2,data_folder="datasets"):
  output_folder="tfrecord_datasets"
  if os.path.exists(output_folder):
    # check if there exists any files in the folder
    if any(os.listdir(output_folder)):
      while True:
        answer = input("tfrecords dataset found. Would you like to overwrite it?\n(pick y if you changed your dataset size, otherwise pick n) y/n:\n")
        answer = answer.lower().strip()
        if answer == "n":
          # return the class names and total_size if the answer is no
          data_files = [file for file in os.listdir(data_folder) if file.endswith(".npy")]
          data_sets = [np.load(os.path.join(data_folder, file)) for file in data_files]
          class_names = [name.replace("full_numy_bitmap_","",1).replace("_","").replace(".npy","") for name in data_files]
          if size_per_class=="full":
            total_size = sum([data.shape[0] for data in data_sets])
          elif size_per_class=="ufull":
            total_size= min([data.shape[0] for data in data_sets]) * len(data_files)
          else:
            total_size = size_per_class * len(data_files)
          return class_names,total_size
        elif answer == "y":
          # break the loop and continue if the answer is yes
          break


  # get the dataset files
  data_files = [file for file in os.listdir(data_folder) if file.endswith(".npy")]
  num_classes = len(data_files)
  # load the dataset files
  data_sets = [np.load(os.path.join(data_folder, file)) for file in data_files]
  # cut the dataset files
  if size_per_class == 'full':
    data=[data_set for data_set in data_sets]
    sizes = [data_set.shape[0] for data_set in data_sets]
    labels=[np.full((sizes[i],),i) for i in range(0,len(data_sets))]
  elif size_per_class == 'ufull':
    min_size = min([data_set.shape[0] for data_set in data_sets])
    data = [data_set[:min_size] for data_set in data_sets]
    labels=[np.full((min_size,),i) for i in range(0,len(data_sets))]
  else:
    data=[data_set[:size_per_class] for data_set in data_sets]
    labels=[np.full((size_per_class,),i) for i in range(0,len(data_sets))]
  # concatenate the lists of arrays to make numpy arrays
  data = np.concatenate(data,axis=0)
  labels = np.concatenate(labels, axis=0)
  total_size = data.shape[0]
  # test percentage of the data
  test_per = round(training_to_test_ratio,ndigits=2)
  os.makedirs(output_folder,exist_ok=True)
  split_amount = int(total_size // split_size)
  train_index =int( (1- test_per) * split_size )

  for i in tqdm(range(split_amount), desc="Writing the data into tfrecords files: "):
      # split the data to 100 samples
      data_split, labels_split = data[i*split_size:(i+1)*split_size], labels[i*split_size:(i+1)*split_size]
      data_split, labels_split = sklearn.utils.shuffle(data_split, labels_split,random_state=1) # (random state makes the shuffle determined)
      # write the dataset into tfrecords files
      numpy_to_tfrecord(data_split[:train_index], labels_split[:train_index], f"{output_folder}/train_data_shard{i}.tfrecords")
      numpy_to_tfrecord(data_split[train_index:], labels_split[train_index:], f"{output_folder}/val_data_shard{i}.tfrecords")
  # Write the remaining data
  if not total_size % split_size:
    remaining_data, remaining_labels = data[split_amount*split_size:], labels[split_amount*split_size:]
    amount = remaining_data.shape[0]
    train_amount = amount - int(amount*test_per)
    numpy_to_tfrecord(remaining_data[:train_amount], remaining_labels[:train_amount], f"{output_folder}/train_data_shard{split_amount}.tfrecords")
    numpy_to_tfrecord(data_split[train_amount:], labels_split[train_amount:], f"{output_folder}/val_data_shard{split_amount}.tfrecords")

  class_names = [name.replace("full_numy_bitmap_","",1).replace("_","").replace(".npy","") for name in data_files]
  return class_names,total_size

# a class to make tensorboard work with model.fit multiple times
class FixedTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs):
        super(FixedTensorBoard, self).__init__(*args, **kwargs)
        self.epoch_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super(FixedTensorBoard, self).on_epoch_end(self.epoch_counter, logs)
        self.epoch_counter += 1

def train(param_dict):
    split_size = 100 # the split size of the data when converting it to tfrecords format.
    # Changing it to a number which is not a multiple of 100 may result in more training data and less validation data than expected.
    class_names,total_size = write_data(param_dict['size_per_class'], split_size, param_dict['training_to_test_ratio'])
    num_classes = len(class_names)
    # create parameters based on the param_dict
    size_per_class = param_dict['size_per_class']
    # test percentage of the data
    test_per = param_dict['training_to_test_ratio']
    # train percentage of the data
    train_per = 1 - test_per
    # Others
    batch_size = param_dict['batch_size']
    model_dir = param_dict['model_dir']
    model_name = param_dict['model_name']
    save_param_dict = param_dict['save_param_dict']
    save_model_checkpoints = param_dict['save_model_checkpoints']
    checkpoints_interval = param_dict['checkpoints_interval']
    load_model = param_dict['load_model']
    if load_model:
      start_from_epoch = param_dict['start_from_epoch']
    else:
      start_from_epoch = 0

    # Create the model according to the parameters
    if not load_model:
      model = create_model(param_dict,num_classes)
    else:
      model = keras.models.load_model(f"{model_dir}/{model_name}.h5")
    def _parse_function(proto):
        keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                            'label': tf.io.FixedLenFeature([], tf.int64)}
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)

        # Convert the image data from string to tensor
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [28,28,1])  / 255.0 # Reshape and normalize the entries of the tensor to be in [0,1]

        label = parsed_features['label']

        return image, label

    # Create a tensorboard callback to analyze the training
    fixed_tensorboard = FixedTensorBoard(log_dir=model_dir)
    split_amount = total_size // split_size

    if not total_size % split_size:
      split_amount = split_amount + 1
    train_tfrecord_files = [f"tfrecord_datasets/train_data_shard{i}.tfrecords" for i in range(split_amount)]
    val_tfrecord_files = [f"tfrecord_datasets/val_data_shard{i}.tfrecords" for i in range(split_amount)]

    def create_train_data():
        # Shuffle the order of the shard filenames at the beginning of each epoch
        np.random.shuffle(train_tfrecord_files)

        # Create a dataset from the shuffled filenames
        dataset = tf.data.Dataset.from_tensor_slices(train_tfrecord_files)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename).map(_parse_function),
            cycle_length=20,  # Number of files to read at once
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Shuffle the examples in each shard using a shuffle buffer
        dataset = dataset.shuffle(buffer_size=split_size*2)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset
    def create_val_data():

        dataset = tf.data.Dataset.from_tensor_slices(val_tfrecord_files)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename).map(_parse_function),
            cycle_length=20,  # Number of files to read at once
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.batch(batch_size)

        return dataset

    # Train the model
    steps_per_epoch = train_per * total_size  // batch_size
    val_data = create_val_data()
    val_steps = test_per * total_size // batch_size

    step_counter = 0
    for epoch in range(param_dict['total_epochs']):
        train_dataset = create_train_data()
        model.fit(train_dataset, steps_per_epoch=steps_per_epoch, validation_data=val_data, validation_steps=val_steps, epochs=1, callbacks=[fixed_tensorboard])
        print(f"Epoch {epoch+1+start_from_epoch}/{param_dict['total_epochs']+start_from_epoch} done!")
        if save_model_checkpoints and epoch+1% checkpoints_interval==0:
          print("Saving checkpoint...")
          model.save(f"{model_dir}/e{epoch+1+start_from_epoch}.h5")
          print("Checkpoint saved!")

    # Save the model
    model.save(f'{model_dir}/{model_name}.h5')
    # Save the param_dict
    if save_param_dict:
        with open(f"{model_dir}/{model_name}_conf.txt","w") as f:
            f.write(str(param_dict))

def test(model_name):
    # Import the SketchApp class and the Preprocess function
    from draw_board import SketchApp, Preprocess
    model = keras.models.load_model(model_name) # load the model
    # class names
    class_names = ['airplane', 'alarm clock', 'ambulance', 'angel',
                   'anvil', 'apple', 'arm', 'axe',
                   'backpack', 'banana', 'bandage', 'barn',
                   'bat', 'boomerang', 'bowtie', 'The Eiffel Tower']
    # sort the classes alphabethically with disregard to upper/lower cases
    class_names = sorted(class_names,key=str.casefold)
    # the predict function which takes an image and returns a text
    def predict(image):
        # preprocess the image
        pre= Preprocess(image)
        # get the prediction matrix (vector in this case)
        prediction_matrix = model.predict(pre)
        # get the index of the class with the most probablity
        prediction_index = np.argmax(prediction_matrix[0])
        
        prediction = class_names[prediction_index] + f" with {prediction_matrix[0][prediction_index]*100}% certainty"

        return prediction
    # initialize and run the canvas
    canvas = SketchApp(predict)
    canvas.run()

param_dict= {
    # Model parameters
    "conv_layers": [(128,(3,3)),(32,(5,5))], # Enter the amount of filters and kernel size as a tuple.
    "pooling_layers": [None,((2,2),(2,2))], # Enter the pooling size and stride amount as a tuple. Add None to skip a layer.
    "conv_dropout_layers": [None,0.2] ,# Leave empty if you do not want dropout layers, enter the dropout rate otherwise. Add None to skip a layer.
    "dense_layers": [512,128], # Enter the amount of neurons in the dense layers
    "dropout_layers": [0.4,0.3], # Leave empty if you do not want dropout layers, enter the dropout rate otherwise. Add None to skip a layer.

    # Training Parameters
    "size_per_class": 'ufull'  , # the amount of samples per class, put "full" to use the full dataset (not recommended)
    # put 'ufull' to use as much of the data while keeping a uniform amount of samples per class
    "training_to_test_ratio": 0.2 , # Training data to testing data ratio. Put 0 to train the model on all the data
    # This parameter automatically ignores anything after the second decimal point
    "batch_size": 64 , # Batch size
    "total_epochs": 30 , # Amount of epochs

    # Saving parameters
    "model_dir": "sketch_classifier_model" , # The directory in which the model and related files will be saved
    "model_name": "sketch_classifier" , # Name of the model
    "load_model": False, # Whether to load the model from a file (specified by the model_name and model_dir parameter) or not
    "start_from_epoch": 0, # If load_model is set to true, what epoch should it start at (only changes the naming of checkpoints)
    "save_param_dict": True , # Whether or not to save the param dict as a file in the model directory
    "save_model_checkpoints": True , # Whether or not to save checkpoints of the model (for early stopping)
    "checkpoints_interval": 1 , # The amount of epochs to wait before saving a checkpoint (ignore this if you chose False in save_model_checkpoints)
    }

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "train":
        try:
            train(param_dict)
            input("Training is done. Press anything to close this window.")
        except Exception as e:
            print("Error:\n"+str(e))
            input("Press anything to close this window.")
        exit()
    elif arg == "test":
        model_name = sys.argv[2]
        try:
            test(model_name)
        except Exception as e:
            print("Error:\n"+str(e))
            input("Press anything to close this window.")
        exit()
 
    
