#########
# tf.keras with tf.distribute for the inclusive classifier
# 
# tested with tf 2.0.0-beta1
########

## Configuration
import os

number_workers=int(os.environ['TOT_WORKERS'])
worker_number=os.environ['WORKER_NUMBER']

# Data paths
train_data = "s3://datasets/tf/train/" 
test_data = "s3://datasets/tf/test/"
model_output_path = "s3://luca/results/"

# Tunables
batch_size = 128 * number_workers
validation_batch_size = 1024
num_epochs = 8

###########

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Masking, Dense, Activation, GRU, Dropout, concatenate
import json

## GRU branch
gru_input = Input(shape=(801,19), name='gru_input')
a = gru_input
a = Masking(mask_value=0.0)(a)
a = GRU(units=50, activation='tanh')(a)
gruBranch = Dropout(0.2)(a)

hlf_input = Input(shape=(14), name='hlf_input')
b = hlf_input
hlfBranch = Dropout(0.2)(b)

c = concatenate([gruBranch, hlfBranch])
c = Dense(25, activation='relu')(c)
output = Dense(3, activation='softmax')(c)
    
model = Model(inputs=[gru_input, hlf_input], outputs=output)
    
## Compile model
optimizer = 'Adam'
loss = 'categorical_crossentropy'
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"] )

# test dataset 
files_test_dataset = tf.data.Dataset.list_files(test_data + "part-r-*")
# training dataset 
files_train_dataset = tf.data.Dataset.list_files(train_data + "part-r-*")

# tunable
num_parallel_reads=tf.data.experimental.AUTOTUNE # TF2.0
# num_parallel_reads=8

test_dataset = files_test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).interleave(
    tf.data.TFRecordDataset, 
    cycle_length=num_parallel_reads,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = files_train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).interleave(
    tf.data.TFRecordDataset, cycle_length=num_parallel_reads,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Function to decode TF records into the required features and labels
def decode(serialized_example):
    deser_features = tf.io.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'HLF_input': tf.io.FixedLenFeature((14), tf.float32),
          'GRU_input': tf.io.FixedLenFeature((801,19), tf.float32),
          'encoded_label': tf.io.FixedLenFeature((3), tf.float32),
          })
    return((deser_features['GRU_input'], deser_features['HLF_input']), deser_features['encoded_label'])

parsed_test_dataset=test_dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
parsed_train_dataset=train_dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train=parsed_train_dataset.repeat().batch(batch_size)

test=parsed_test_dataset.repeat().batch(validation_batch_size)

num_train_samples=3426083   # there are 3426083 samples in the training dataset
steps_per_epoch=num_train_samples//batch_size

num_test_samples=856090 # there are 856090 samples in the test dataset
validation_steps=num_test_samples//validation_batch_size  

# Example callback to write logs for tensorboard
# callbacks = [ tf.keras.callbacks.TensorBoard(log_dir = "./logs") ]
# Example callback for model checkpointing
# callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = model_output_path + "keras-ckpt")]
callbacks = [] # no callbacks

history = model.fit(train, steps_per_epoch=steps_per_epoch, \
                    validation_data=test, validation_steps=validation_steps, \
                    epochs=num_epochs, callbacks=callbacks, verbose=1)

##Training is finished

model.evaluate(test, steps=validation_steps)

model_full_path=model_output_path + "mymodel" + str(worker_number) + ".h5"
print("Training finished, now saving the model in h5 format to: " + model_full_path)

model.save(model_full_path, save_format="h5")

# TensorFlow 2.0
model_full_path=model_output_path + "mymodel" + str(worker_number) + ".tf"
print("..saving the model in tf format (TF 2.0) to: " + model_full_path)
tf.keras.models.save_model(model, PATH+"mymodel" + ".tf", save_format='tf')

