# ANN_Reforming_Modeling
This is the official GitHub repository of the paper "Improving the robustness of the hydrocarbon steam reforming kinetic models based on artificial neural networks".

## Description 
The main purpose of this repository is to share the code relative to the In-Silico data generation step and the kinetic model discrimination through an Artificial Neural Network. This code was fully written in Python using an Anaconda virtual environment.

## Dependancies
The Anaconda enviroment used in this work mainly requires commonly used packages like Numpy, Scipy, Pandas, Scikit-Learn, Tensorflow or Keras. To replicate the environment, download the provided [ANN_Naphtha_Reforming.yml](environment.yml) file and type the following command:
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
- params_dict: 
- instances_per_model:
- models: 
- sigmar:
- sigmac:
- distribution: 

The generated data is provided 


## Authorship

## Acknowledgments

## Citations
