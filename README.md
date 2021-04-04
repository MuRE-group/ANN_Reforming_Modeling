# ANN_Reforming_Modeling
This is the official GitHub repository of the paper "Improving the robustness of the hydrocarbon steam reforming kinetic models based on artificial neural networks".

## Description
The main purpose of this repository is to share the code relative to the In-Silico data generation step and the kinetic model discrimination through an Artificial Neural Network. This code was fully written in Python using an Anaconda virtual environment.

## Dependancies
The Anaconda enviroment used in this work mainly requires commonly used packages like Numpy, Scipy, Pandas, Scikit-Learn, Tensorflow or Keras. To replicate the environment, download the provided [ANN_Naphtha_Reforming.yml](ANN_Naphtha_Reforming.yml) file and type the following command:
```
conda env create -f ANN_Naphtha_Reforming.yml
```
For more information, refer to [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## Files
The repository mainly contains 1 spreadsheet, 2 python files and 2 folders:
- ```Raw_Data.xlsx```
- ```In_Silico.py```
- ```ANN_In_Silico.py```
- ```In_Silico/```
- ```Trained_Models/```

### ```Raw_Data.xlsx```
It is a spreadsheet containing the experimental data used, not only for the model discrimination, but also for the parameter estimation.

### ```In_Silico.py```
This file corresponds to the In_Silico data generation step and requires the following user-specified inputs:
- ***params_dict***: A dictionary with the kinetic parameters (keys) and a numpy array with the lower and upper boundaries (values).
- ***instances_per_model***: Number of instances per model for which the conservation equation should be solved.
- ***models***: A list of the models for which the conservation equation should be solved.
- ***sigmar***: Parameter to tune the Gaussian noise that can be added to the generated data.
- ***sigmac***: Parameter to tune the Gaussian noise that can be added to the generated data.
- ***distribution***: If true, displays a distribution of the kinetic parameter selection.

The generated data is provided both in .csv (```Data_in_silico_*instances_per_model*.csv```) and .xlsx (```Data_in_silico_*instances_per_model*.xlsx```) formats. Also, a text summary (```README_In_Silico.txt```) is provided.

### ```In_Silico/```
This folder contains several folders underneath which contain the data generated in-silico for all the cases reported in the paper. In each of the subfolders, data corresponding to same kinetic parameter domain without noise is stored, varying the number of instances per model (50, 125, 250, 500, 750, 1000) from one to another. Each of the subfolders contains the output of ```In_Silico.py```:
- ```Data_in_silico_*instances_per_model*.csv```
- ```Data_in_silico_*instances_per_model*.xlsx```
- ```README_In_Silico.txt```

### ```Trained_Models/```
This folder also contains several folders, each of them corresponding to the same cases present in ```In_Silico/. Each subfolder contains the output of ANN_In_Silico.py```:
- The files ```Gridsearch.csv``` and ```Gridsearch.xlsx``` contain the complete results from the grid search for the ANN. 
- The model with the best performing hyperparameters is provided in both json (```best_model.json```) and h5 (```best_model.h5```) formats, as suggested by [Tensorflow](https://www.tensorflow.org/guide/keras/save_and_serialize). 
- ```README_In_Silico.txt```
- ```README_Best_Training.txt```

## Authorship

## Acknowledgments

## Citations
