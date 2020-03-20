## Training using tf.keras and tf.distribute parallelized on Kubernetes clusters for CPU and GPU training

This folder contains code for training the Inclusive Classifier with tf.keras in distributed mode using Kubernetes resource.
It is intended to run with the helper script [tf-spawner](https://github.com/cerndb/tf-spawner)
- `4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py` 
- `4.3a_InclusiveClassifier_WorkerCode_tuned_NOcached_learningAdaptive.py`


**How to use:**
- Download TF-Spawner: `git clone https://github.com/cerndb/tf-spawner`
- Install the dependencies: `pip3 install kubernetes` 
- Set up you Kubernetes environment (if needed): `export KUBECONFIG=<path_to_kubectl config file`
- Copy the data, for example on a S3-compatible filesystem, and set the required environment variables,
 for example edit the file `examples/envfile.example` with the
  ```
  S3_ENDPOINT=
  AWS_ACCESS_KEY_ID=
  AWS_SECRET_ACCESS_KEY=
  AWS_LOG_LEVEL=3
  ```
- Edit the configurable variables in the training script
  - notably: "Data paths" variables, as `PATH="s3://sparktfdata/"`

**Run distributed training on Kubernets with TF-Spawner**
 
- Run on CPU as in this example (with 10 workers):
  - `./tf-spawner -w 10 -i tensorflow/tensorflow:2.0.1-py3 -e examples/envfile.example <PATH>/4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py`

- When training using GPU resources on Kubernetes use this example (with 10 GPU nodes):
  - `./tf-spawner -w 10 -i tensorflow/tensorflow:2.0.1-gpu-py3 --pod-file pod-gpu.yaml -e examples/envfile.example <PATH>/4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py`

**Notes:**
The training scripts provided in this folder have been tested on Kubernetes clusters for CPU and GPU resources, using TF 2.0.1.  
The scripts use:
- tf.distribute strategy with MultiWorkerMirroredStrategy to parallelize the training.
- tf.data to read the training and test dataset, from files in TFRecord format (see also [data folder](../Data)).
