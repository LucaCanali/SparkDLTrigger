This folder contains code for training the Inclusive classifier with tf.keras in distributed mode.
tf.distrubute strategy with MultiWorkerMirroredStrategy is used to parallelize the training.
tf.data is used to read the data in TFRecord format.

- MultiWorker_Notebooks: distributed training and model performance metrics visualization using notebooks
- DataPrep_extract_and_convert_Full_Dataset_TFRecord.scala: data conversion from Apache Parquet to TFRecord
- MultiWorker_PythonCode: distributed training using "manual" Python code, suitable for local node testing
- MultiWorker_Kubernetes: Distributed training on Kubernets cluster, using the custom TF-Spawner tool 

