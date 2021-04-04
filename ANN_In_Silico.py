import numpy  as np
import pandas as pd

import pickle

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import tensorflow as tf

np.random.seed(10)
tf.random.set_seed(10)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier

# Load data for Naphtha Reforming from CSV format
instances_per_model = 50
df = pd.read_csv('Data_in_silico' + '_' + str(instances_per_model) + '.csv')
df = df.set_index(df.columns[0])
df.index.names = ['No.']
df.info()

global models
#Load model list
filename = 'model_list' + '_' + str(instances_per_model) + '.sav'
file = open(filename, 'rb')
models = pickle.load(file)
file.close()

#Label data
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

# encode class values as integers
encoder = LabelEncoder()
y_one_hot = np_utils.to_categorical(encoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, random_state = 10, shuffle = True, test_size = 0.2)

#Feature scaling is required
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Define the net
def in_silico_ANN(units1, activation1, input_dim, dropout1, optimizer, loss):

    #Create keras ANN model
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units = units1, activation = activation1, input_dim = input_dim),
            tf.keras.layers.Dropout(dropout1),
            tf.keras.layers.Dense(units = len(models), activation = 'softmax')])
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    return model

#hyperparameter dictionary for KerasClassifier
hypar = {'units1'       : [100, 250, 500, 750, 1000, 1500],
         'activation1'  : ['relu', 'elu', 'tanh'],
         'input_dim'    : [X_train_scaled.shape[1]],
         'dropout1'     : [0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45],
         'optimizer'    : ['adam'],
         'loss'         : ['categorical_crossentropy'],
         'epochs'       : [500, 1500, 3000, 5000],
         'batch_size'   : [64, 128, 512]}

ANN = KerasClassifier(build_fn = in_silico_ANN, verbose = 0)

#Gridsearch
search = GridSearchCV(ANN, hypar,
                      cv = KFold(n_splits = 10, shuffle = True, random_state = 10),
                      n_jobs = -1, verbose = 0)

search.fit(X_train_scaled, y_train)

#Export the search
results = pd.DataFrame(search.cv_results_)
results.to_csv('Gridsearch.csv')
results.to_excel('Gridsearch.xlsx')

#Export the best model to JSON and HDF5
best_model = in_silico_ANN(search.best_params_['units1'],    search.best_params_['activation1'],
                           search.best_params_['input_dim'], search.best_params_['dropout1'],
                           search.best_params_['optimizer'], search.best_params_['loss'])

best_model.fit(X_train_scaled, y_train, epochs = search.best_params_['epochs'],
                                        batch_size = search.best_params_['batch_size'], verbose = 0)

with open('best_model.json', 'w') as json_file:
    json_file.write(best_model.to_json())

best_model.save_weights("best_model.h5")

#Make predictions
y_pred_train = search.predict(X_train_scaled)
y_pred_test = search.predict(X_test_scaled)

#Read data from experimental excel
df_exp  = pd.read_excel('Raw_Data.xlsx')

df_predict = df_exp[(df_exp[df_exp.columns[2]] > 0.10) & (df_exp[df_exp.columns[2]] < 0.90)]
df_predict = df_predict.iloc[:, 2:-1].values
df_predict = np.concatenate(df_predict).ravel()
df_predict = df_predict.reshape(1, df_predict.shape[0])

Data_scaled = scaler.transform(df_predict)
prob_model = search.predict_proba(Data_scaled)

#Write the predicitons

text_file = open('README_Best_Training.txt', 'a')

with open('README_Best_Training.txt','w') as file:
    file.write('Best fitting hyperparameters\n\n')
    for x,y in dict(search.best_params_).items():
        file.write('{}:\t{}\n'.format(x, y).expandtabs(20))
    file.write('\nAccuracy tests \n')
    file.write('\nTraining phase:\t {0:.3f}\n'.format((accuracy_score(np.argmax(y_train, axis = 1), y_pred_train))))
    file.write('Testing phase:\t {0:.3f}\n'.format((accuracy_score(np.argmax(y_test, axis = 1), y_pred_test))))
    file.write('\nConfusion matrix\n')
    file.write('\n{0}\n'.format(confusion_matrix(np.argmax(y_test, axis = 1), y_pred_test)))
    file.write('\nClassification report\n')
    file.write('\n{0}\n'.format(classification_report(np.argmax(y_test, axis = 1), y_pred_test, target_names = models)))
    file.write('\nProbabilities (%) for experimental data\n')
    file.write('\n')
    for i in range(len(models)):
        file.write('{}:\t{:.2f}\n'.format(models[i], prob_model[0][i]*100).expandtabs(70))
