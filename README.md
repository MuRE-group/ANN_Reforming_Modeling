# ANN-based kinetic model discrimination 
This is the official GitHub repository of the paper "Improving the robustness of the hydrocarbon steam reforming kinetic models based on artificial neural networks". This repository contains the code relative to the *in-silico* data generation and kinetic model discrimination steps, fully written in Python using an Anaconda virtual environment.

## Setting up the environment
The Anaconda enviroment used in this work mainly requires commonly used packages like Numpy, Scipy, Pandas, Scikit-Learn, Tensorflow or Keras. To replicate the environment, download the provided ```ANN_Naphtha_Reforming.yml``` file and type the following command:
```
conda env create -f ANN_Naphtha_Reforming.yml
```
For more information, refer to [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## Files
This repository mainly contains 1 spreadsheet, 2 python files and 2 folders:
- ```Raw_Data.xlsx```
- ```In_Silico.py```
- ```ANN_In_Silico.py```
- ```In_Silico/```
- ```Trained_Models/```

The workflow followed in this paper is as follows:

![Workflow](/images/Workflow.jpeg)

### ```Raw_Data.xlsx```
A spreadsheet containing the raw experimental data used, not only for the model discrimination, but also for the parameter estimation.

### ```In_Silico.py```
A python script for the in-silico data generation step that requires the following user-specified inputs:
- ***params_dict***: A dictionary with the kinetic parameters (keys) and a numpy array with the lower and upper boundaries (values).
- ***instances_per_model***: Number of instances per model for which the conservation equation should be solved.
- ***models***: A list of the models for which the conservation equation should be solved.
- ***sigmar***: Parameter to tune the Gaussian noise that can be added to the generated data.
- ***sigmac***: Parameter to tune the Gaussian noise that can be added to the generated data.
- ***distribution***: If true, displays a distribution of the kinetic parameter selection.

This file has the following dependencies:
- ```Raw_Data.xlsx```

The generated data is provided both in .csv (```Data_in_silico_*instances_per_model*.csv```) and .xlsx (```Data_in_silico_*instances_per_model*.xlsx```) formats. A list with the names of the models of choice (```model_list_*instances_per_model*.sav```) is also given as output. A text summary (```README_In_Silico.txt```) is provided.

### ```ANN_In_Silico.py```
This file corresponds to the training of the Artificil Neural Network and requires the following user-specified inputs:
- ***instances_per_model***: Number of instances per model for which the conservation equation has been solved.
- ***hypar***: A dictionary with the hyperparameter (keys) and a numpy array with their domains in which the gridsearch should be performed (values).

This file has the following dependencies:
- ```Raw_Data.xlsx```
- ```Data_in_silico_*instances_per_model*.csv```
- ```model_list_*instances_per_model*.sav```

The files ```Gridsearch.csv``` and ```Gridsearch.xlsx``` contain the complete results from the grid search for the ANN. The model with the best performing hyperparameters is provided in both json (```best_model.json```) and h5 (```best_model.h5```) formats, as suggested by [Tensorflow](https://www.tensorflow.org/guide/keras/save_and_serialize). A text summary of the ANN training outcome ```README_Best_Training.txt``` is also provided.

### ```In_Silico/```
This folder contains several (compressed) folders underneath which contain the data generated in-silico for most of the cases reported in the paper. In each of the subfolders, data corresponding to same kinetic parameter domain without noise is stored, varying the number of instances per model (50, 125, 250, 500) from one to another. Each subfolder contains the output of ```In_Silico.py```:
- ```Data_in_silico_*instances_per_model*```
- ```model_list_*instances_per_model*.sav```
- ```README_In_Silico.txt```

Because of the large size of some .csv files, these have been pickled and therefore, have to be unpickled in order to be used. For instance, to unplickle the data corresponding to 50 instances (```Data_in_silico_50```) into ```Data_in_silico_50.csv```, the following commands should be employed:

```
with open('Data_in_silico_50', 'rb') as picklefile:
       df = pickle.load(picklefile)
       df.to_csv('Data_in_silico_50.csv')
```

### ```Trained_Models/```
This folder also contains several folders, each of them corresponding to the same cases present in ```In_Silico/```. Each subfolder stores the output of ```ANN_In_Silico.py```:
- ```Gridsearch.csv``` 
- ```Gridsearch.xlsx```
- ```best_model.json```
- ```best_model.h5```
- ```README_Best_Training.txt``` 

## Acknowledgements
The authors gratefully acknowledge financial support, resources and facilities provided by the King Abdullah University of Science and Technology (KAUST). 

## Citations

## Contact 
This work has been carried out within the Multiscale Reaction Engineering ([MuRE](https://mure.kaust.edu.sa)) group, led by Dr. Pedro Castaño, in the KAUST Catalysis Center (Saudi Arabia).
