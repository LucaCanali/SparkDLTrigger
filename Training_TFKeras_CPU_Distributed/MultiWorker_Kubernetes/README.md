This folder contains code for training the Inclusive classifier with tf.keras in distributed mode using Kubernetes resources  
tf.distrubute strategy with MultiWorkerMirroredStrategy is used to parallelize the training.
tf.data is used to read the data in TFRecord format.

**How to use:**  
- Download TF-Spawner: `git clone https://github.com/cerndb/tf-spawner`
- Copy `Train_Inclusive_Classifier.py` into TF-Spawner directory
- Set up you environment for Kubernetes: `export KUBECONFIG=<path_to_kubectl config file`
- Copy the data, for example on a S3-compatible filesystem and set the required variables:
  ```
  export AWS_ACCESS_KEY_ID="...."
  export AWS_ACCESS_KEY_ID="...."
  export AWS_ACCESS_KEY_ID="...." (example "s3.cern.ch")
  ```
- Edit the configurable variables in `Train_Inclusive_Classifier.py`
  - notably: "Data paths" variables

- Run distributed training on Kubernets with TF-Spawner as in this example:
  - `python launch.py -w 32 Train_Inclusive_Classifier.py`

