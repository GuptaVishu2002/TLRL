# TLRL

This repository contains the code for performing transfer learning + representation learning to predict materials properties using elemental fractions (EF), physical attributes (PA) or extracted features as the model input. The code provides the following functions:

* Train a Neural Network model on a given dataset.
* Use a pre-trained Neural Network model to perform transfer learning + representation learning on a given target dataset.
* Predict material properties of new compounds with a new pre-trained Neural Network model.

It is recommended to train large dataset (e.g. OQMD, MP) from scratch (SC) and small datasets (DFT-computed or experimental datasets) using transfer learning + representation learning methods.


## Installation Requirements

The basic requirement for using the files are a Python 3.6.3 Jupyter environment with the packages listed in `requirements.txt`. It is advisable to create an virtual environment with the correct dependencies.

The work related experiments was performed on Linux Fedora 7.9 Maipo. The code should be able to work on other Operating Systems as well but it has not been tested elsewhere.

## Source Files
  
Here is a brief description about the folder content:

* [`neuralnetwork`](./neuralnetwork): code for training Neural Network model from scratch or using a pretrained Neural Network model to perform transfer learning + representation learning.

* [`representation`](./representation): Jupyter Notebook to perform feature extraction from a specific layer of pre-trained ElemNet model. We have also provided the code to convert chemical formula of a compound into elemental fractions and physical attributes.

* [`prediction`](./prediction): Jupyter Notebook to perform prediction using the pre-trained ElemNet model.

## Example Use

### Create a customized dataset

To use different representations (such as elemental fractions, physical attributes or extracted features) as input to ElemNet, you will need to create a customized dataset using a `.csv` file. You can prepare a customized dataset in the following manner:

1. Prepare a `.csv` file which contain two columns. The first column contains the compound, and the second column contains the value of target property as shown below:
 
| pretty_comp | target property |
| ----------- | --------------- |
| KH2N        | -0.40           |
| NiTeO4      | -0.82           |

2. Use the Jupyter Notebook in [`representation`](./representation) folder to pre-process and convert the first column of the `.csv` file into elemental fraction, physical attributes or to extract features using a pre-trained Neural Network model. Above example when converted into elemental fraction becomes as follows: 

| pretty_comp |  H  | ... | Pu | target property |
| ----------- | --- | --- | -- | --------------- |
| KH2N        | 0.5 | ... | 0  | -0.40           |
| NiTeO4      | 0   | ... | 0  | -0.82           |

3. Split the customized dataset into train, validation and test set. We have used <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">train_test_split</a> function of the sklearn library with a random seed of 1234567 to perform the train\validation\test split in our work.

We have provided an example of customized datasets in the repository: `data/sample`. Here we have converted the first column of the `.csv` file into elemental fractions. Note that this is required for both training and predicting. 

### Run Neural Network model

The code to run the Neural Network model is provided in the [`neuralnetwork`](./neuralnetwork) folder. In order to run the model you can pass a sample config file to the dl_regressors_tf2.py from inside of your neuralnetwork directory:

`python dl_regressors_tf2.py --config_file sample/sample-run_example_tf2.config` (without TL+RL)
`python dl_regressors_tf2_tlnewinput.py --config_file sample/sample-run_example_tf2.config` (with TL+RL)

The config file defines all the related hyperparameters associated with the model training and model testing such as loss_type, training_data_path, val_data_path, test_data_path, label, input_type, etc. 

For setting pre-trained weight, you need to set it in 'model_path' [e.g. `model/sample_model`]. 

To perform TL+RL using dl_regressors_tf2_tlnewinput.py set the 'pretrainedmodel_input' hyperparameter of the config file to the inputs used to train the pre-trained model

To add customized input_type, please make changes to the `data_utils.py` as follows:

1. Add a new array with the required columns (preferably near the place where other arrays are defined). For example:

|  a  |  b  |  c  |  d  | pred |
| --- | --- | --- | --- | ---- |
| 0.1 | 0.3 | 0.5 | 0.7 | 0.9  |
| 0.2 | 0.4 | 0.6 | 0.8 | 1.0  |

   If you have the following `.csv` file where you have to use columns a, b, c, d to predict pred, you can add `new_input = ['a','b','c','d']` to the file.

2. Add the array variable to the `input_atts` dictionary so that it can be used with input_type of the config (and pred used with label). For example:

   `input_atts = {'new_input':new_input, 'elements':elements, ... , 'input32':input32}`
   
   
After training, you will get the following files:

* The output log from the training will be saved in the [`log`](./neuralnetwork/log) folder as `log/sample-run_example_tf2.log` file.

* The trained model will be saved in [`model`](./neuralnetwork/model) folder as `model/sample-run_example_tf2.h5` and `model/sample-run_example_tf2.json`. `.h5` file contains the weights values and `.json` contains the model's architecture. </br> </br>
We also save the model in a newer version of the TensorFlow SavedModel format in [`sample`](./neuralnetwork/sample) folder as `sample-run_example_tf2` folder. The model architecture, and training configuration (including the optimizer, losses, and metrics) are stored in saved_model.pb. The weights are saved in the variables/ directory. For more information <a href="https://www.tensorflow.org/guide/keras/save_and_serialize">see here</a> </br> </br>


The above command runs a default task with an early stopping of 200 epochs on small dataset of target property (formation energy). This sample task can be used without any changes so that the user can get an idea of how the Neural Network model works. The sample task should take about 1-2 minute on an average when a GPU is available and give a test set MAE of 0.29-0.32 eV after the model training is finished.

Note: Your <a href="https://machinelearningmastery.com/different-results-each-time-in-machine-learning/">results may vary</a> given the stochastic nature of the algorithm or evaluation procedure, the size of the dataset, the target property to perform the regression modelling for or differences in numerical precision. Consider running the example a few times and compare the average outcome.

## Developer Team

The code was developed by Vishu Gupta from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

## Publication


## Acknowledgements

The open-source implementation of CrossPropertyTL <a href="https://github.com/NU-CUCIS/CrossPropertyTL">here</a> provided significant initial inspiration for the structure of this code-base.

## Disclaimer

The research code shared in this repository is shared without any support or guarantee on its quality. However, please do raise an issue if you find anything wrong and I will try my best to address it.

email: vishugupta2020@u.northwestern.edu

Copyright (C) 2023, Northwestern University.

See COPYRIGHT notice in top-level directory.

## Funding Support

This work was performed under the following financial assistance award 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from Department of Energy (DOE) awards DE-SC0019358, DE-SC0021399, and NSF award CMMI-2053929.
