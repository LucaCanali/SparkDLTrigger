# Data

This directory contains links to download the datasets used by
["Machine Learning Pipelines with Modern Big Data Tools for High Energy Physics"](https://arxiv.org/abs/1909.10389), with a short description.  
For the largest datasets (raw data and the output of first step of pre-processing) we have currently uploaded
only representative samples. The full dataset is expected to be made available using CERN Open Data at a later date.
Datasets are made available under the terms of the CC0 waiver.   
Credits: for the original (rawData) dataset to [T.Q.Nguyen *et al.*](https://arxiv.org/abs/1807.0008).  
Datasets for Machine Learning, available in Apache Parquet and TFRecord formats have been produced using the notebooks published in this repository.  
Note: If you have access to CERN computing resources, you can contact the authors to get
more information on where to find the full datasets, that are available both on the Hadoop platform and on CERN EOS storage.

## How to Bulk Download Using wget
This technique can be used to download full data directories (tested on Linux):
```
# Select the data set to download
DATASET_NAME="testUndersampled_HLF_features.parquet"
#DATASET_NAME="testUndersampled.parquet"
#DATASET_NAME="testUndersampled.tfrecord"
wget -r -np -R "index.html*" -e robots=off http://sparkdltrigger.web.cern.ch/sparkdltrigger/$DATASET_NAME

DATASET_NAME="trainUndersampled_HLF_features.parquet"
#DATASET_NAME="trainUndersampled.parquet"
#DATASET_NAME="trainUndersampled.tfrecord"
wget -r -np -R "index.html*" -e robots=off http://sparkdltrigger.web.cern.ch/sparkdltrigger/$DATASET_NAME
```

## HLF Features 
This is the simplest model. It contains an array of 14 "High Level Features" (HLF). The classifier has 3 output classes, labeled from 0 to 1.
The training dataset has 3.4M rows and the training dataset 86K rows.
```
Schema:
 |-- HLF_input: array 
 |    |-- element: double 
 |-- encoded_label: array 
 |    |-- element: double 
```
- HLF features in Apache Parquet format (training and test dataset):
  - 300 MB: [trainUndersampled_HLF_features.parquet](http://sparkdltrigger.web.cern.ch/sparkdltrigger/trainUndersampled_HLF_features.parquet)
  - 75 MB: [testUndersampled_HLF_features.parquet](http://sparkdltrigger.web.cern.ch/sparkdltrigger/testUndersampled_HLF_features.parquet)

- HLF features in TFRecord format:
  - 106 MB: [trainUndersampled_HLF_features.tfrecord](http://sparkdltrigger.web.cern.ch/sparkdltrigger/trainUndersampled_HLF_features.tfrecord)
  - 422 MB: [testUndersampled_HLF_features.tfrecord](http://sparkdltrigger.web.cern.ch/sparkdltrigger/testUndersampled_HLF_features.tfrecord)

## Low Level Features for GRU-based model
This is the complete dataset for training the more complex models (based on GRU): the "Particle Sequence Classifier"
and the "Inclusive Classifier". This dataset is a superset and much larger than the HLF Features dataset described above,
as it contains large arrays of particles used by the GRU model.
The training dataset has 3.4M rows and the test dataset has 86K rows.
HLF_Input are arrays contain 14 elements (high level features). GRU_input are arrays of size (801,19), they contain a
list a 801 particles with 19 "low level" features per particle.
The classifier has 3 output classes, labeled from 0 to 1.
```
Schema:
 |-- hfeatures: vector
 |-- label: long 
 |-- lfeatures: array
 |    |-- element: array
 |    |    |-- element: double
 |-- hfeatures_dense: vector
 |-- encoded_label: vector 
 |-- HLF_input: vector
 |-- GRU_input: array 
 |    |-- element: array
 |    |    |-- element: double
```
- Sample dataset with 2k events in Apache Parquet format:
  - 162 MB: [testUndersampled_2kevents.parquet](http://sparkdltrigger.web.cern.ch/sparkdltrigger/testUndersampled_2kevents.parquet) Contains a sample of the test dataset with all the features, in Apache Parquet format, produced by the filtering and feature engineering steps

### Low Level Features in Apache Parquet format:
  - 255 GB: [trainUndersampled.parquet](http://sparkdltrigger.web.cern.ch/sparkdltrigger/trainUndersampled.parquet)
  - 64 GB:  [testUndersampled.parquet](http://sparkdltrigger.web.cern.ch/sparkdltrigger/testUndersampled.parquet)

### Low Level Features  in TFRecord format 
  - Note, this dataset is derived by the full datasets in Parquet.
    It contains the same number of rows as the Parquet dataset, but only 3 fields, as needed 
    by the Inclusive classifier with Tensorflow: HLF_input, GRU_input and encoded_labels.
  - 195 GB: [trainUndersampled.tfrecord](http://sparkdltrigger.web.cern.ch/sparkdltrigger/trainUndersampled.tfrecord)
  - 49 GB:  [testUndersampled.tfrecord](http://sparkdltrigger.web.cern.ch/sparkdltrigger/testUndersampled.tfrecord)

## Raw Data - SAMPLE
Only a sample of the raw data is provided at present. The full dataset used by this work occupies 4.5TB.
- 14 GB [lepFilter_rawData_SAMPLE](http://sparkdltrigger.web.cern.ch/sparkdltrigger/lepFilter_rawData_SAMPLE)

## Output of the First Data Processing Step - SAMPLE
Only a sample of the data is provided currently, The full datataset occupies 943 GB.
- 6.4 GB [dataIngestion_full_13TeV_SAMPLE](http://sparkdltrigger.web.cern.ch/sparkdltrigger/dataIngestion_full_13TeV_SAMPLE)
