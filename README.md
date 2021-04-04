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
The repository mainly contains 2 python files:
- ```In_Silico.py```
- ```ANN_In_Silico.py```

Additionally, the In-Silico generated data for all the instances reported in the paper (50, 125, 250, 500, 750 and 1000) are included here. The models corresponding to such data are also available here. 

### ```In_Silico.py```
This file requires the following user-specified inputs:
- ***params_dict***: A dictionary with the kinetic parameters (keys) and a numpy array with the lower and upper boundaries (values).
- ***instances_per_model***: Number of instances per model for which the conservation equation should be solved.
- ***models***: A list of the models for which the conservation equation should be solved.
- ***sigmar***: Parameter to tune the Gaussian noise that can be added to the generated data.
- ***sigmac***: Parameter to tune the Gaussian noise that can be added to the generated data.
- ***distribution***: If true, displays a distribution of the kinetic parameter selection.

The generated data (Data_in_silico_*instances_per_model*) is provided both in .csv and .xlsx formats. Also, a text summary (README_In_Silico.txt) is provided.

## Authorship

## Acknowledgments

## Citations
