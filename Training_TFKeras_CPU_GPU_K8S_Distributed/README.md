## Training using tf.keras and tf.distribute parallelized on Kubernetes clusters for CPU and GPU training

This folder contains code for training the Inclusive classifier with tf.keras in distributed mode using Kubernetes resource.
It is intended to run with the helper script [tf-spawner](https://github.com/cerndb/tf-spawner)
- `4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py` 
- `4.3a_InclusiveClassifier_WorkerCode_tuned_NOcached_learningAdaptive.py`


**How to use:**
- Download TF-Spawner: `git clone https://github.com/cerndb/tf-spawner`
- Set up you Kubernetes environment (if needed): `export KUBECONFIG=<path_to_kubectl config file`
- Copy the data, for example on a S3-compatible filesystem and set the required variables:
  ```
  export AWS_ACCESS_KEY_ID="...."
  export AWS_SECRET_ACCESS_KEY="...."
  export AWS_LOG_LEVEL=3
  export S3_ENDPOINT="...." #(example "s3.cern.ch")
  ```
- Edit the configurable variables in the training script
  - notably: "Data paths" variables, as `PATH="s3://sparktfdata/"`

- Run distributed training on Kubernets with TF-Spawner as in this example:
  - `./tf-spawner -w 10 <EDIT_PATH>/4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py`

Notes:
The training scripts provided in this folder have been tested on Kubernetes clusters for CPU and GPU resources, using TF 2.0.1.
The scripts use:
- tf.distrubute strategy with MultiWorkerMirroredStrategy to parallelize the training.
- tf.data to read the training and test dataset, from files in TFRecord format (see also [data folder](../Data)).

