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

![Workflow](/images/Workflow.jpeg)

### ```Raw_Data.xlsx```
A spreadsheet containing the raw experimental data used, not only for the model discrimination, but also for the parameter estimation.

### ```In_Silico.py```
A Python script for the *in-silico* data generation step that requires the following user-specified inputs:
- ***params_dict***: A dictionary with the names of the kinetic parameters (keys) and a np.array with the lower and upper boundaries (values).
- ***instances_per_model***: Number of instances per model for which the conservation equation should be solved.
- ***models***: A list of the models for which the conservation equation should be solved. Rate expressions of the specified models need to be updated in the conservation_eq function.
- ***sigmar***: Parameter defining the Gaussian noise, proportional to the generated data.
- ***sigmac***: Parameter defining the Gaussian noise, constant added to the generated data.
- ***distribution***: If true, displays a distribution of the kinetic parameter sampling.

This file has the following dependencies:
- ```Raw_Data.xlsx```

Generated data is provided both in .csv and .xlsx formats. A list with the names of the models of choice is also given as output, along with a text summary. For example, in the case 50 instaces per model are defined, the output files are the following:  
- ```Data_in_silico_50.csv```
- ```Data_in_silico_50.xlsx```
- ```model_list_50.sav```
- ```README_In_Silico.txt```

### ```ANN_In_Silico.py```
A Python script for the training of the artificial neural network that requires the following user-specified inputs:
- ***instances_per_model***: Number of instances per model for which the conservation equation has been solved.
- ***hypar***: A dictionary with the names of the hyperparameter (keys) and a np.array with their domains in which the gridsearch should be performed (values).

This file has the following dependencies (in the case 50 instaces per model are defined):
- ```Raw_Data.xlsx```
- ```Data_in_silico_50.csv```
- ```model_list_50.sav```

The output files ```Gridsearch.csv``` and ```Gridsearch.xlsx``` contain the complete results from the grid search for the ANN. The model with the best performing hyperparameters is provided in both *JSON* (```best_model.json```) and *HDF5* (```best_model.h5```) formats, as suggested by [Tensorflow](https://www.tensorflow.org/guide/keras/save_and_serialize). A text summary of the ANN training outcome ```README_Best_Training.txt``` is also provided.

### ```In_Silico/```
A folder with several folders underneath which contain the data generated *in-silico* for all the cases reported in the paper. In each of the subfolders, generated data corresponding to the kinetic parameter domain specified in the paper without noise is stored, only varying the number of instances per model (50, 125, 250, 500, 750, 1000) from one to another. Specific information is available in the ```README_In_Silico.txt``` file of each case.

As an example, the subfolder with the output of ```In_Silico.py``` for the 50 instances per model contains:
- ```Data_in_silico_50.h5```
- ```model_list_50.sav```
- ```README_In_Silico.txt```

Because of the large size of the files stemming from the cases with high number of instances per model, neither .xlsx nor  .csv files are provided in the reporsitory. Note that the .xlsx files are not required to run ```ANN_In_Silico.py```. Instead, the .csv files have been compressed to *HDF5* format and thus, have to be uncompressed in order to be used. For example, to uncompress the data corresponding to 50 instances (```Data_in_silico_50.h5```) into ```Data_in_silico_50.csv```, the following commands should be employed:
```
with open('Data_in_silico_50.h5', 'rb'):
    df = pd.read_hdf('./Data_in_silico_50.h5')
    df.to_csv('Data_in_silico_50.csv')
```
Note that that line has pandas library dependency.

### ```Trained_Models/```
A folder containing several folders, each of them corresponding to the same cases present in ```In_Silico/```. Each subfolder stores the output of ```ANN_In_Silico.py```:
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
