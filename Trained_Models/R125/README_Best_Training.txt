Best fitting hyperparameters

activation1:        elu
batch_size:         128
dropout1:           0.3
epochs:             5000
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             100

Accuracy tests 

Training phase:	 0.934
Testing phase:	 0.771

Confusion matrix

[[10  2  0  0  6  0  1]
 [ 7 23  0  0  2  1  0]
 [ 0  0 23  0  0  1  0]
 [ 2  0  0 26  3  0  1]
 [ 4  1  0  1 16  1  3]
 [ 1  0  1  0  0 18  2]
 [ 0  0  0  0  0  0 19]]

Classification report

                                                                   precision    recall  f1-score   support

                          Dual Site Adsorption, Naphtha and Water       0.42      0.53      0.47        19
                          Same Site Adsorption, Naphtha and Water       0.88      0.70      0.78        33
             Same Site Dissociative Adsorption, Naphtha and Water       0.96      0.96      0.96        24
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal       0.96      0.81      0.88        32
                                         Single Adsorption, Water       0.59      0.62      0.60        26
                            Single Dissociative Adsorption, Water       0.86      0.82      0.84        22
               Single Dissociative Adsorption, Water, Eley-Rideal       0.73      1.00      0.84        19

                                                         accuracy                           0.77       175
                                                        macro avg       0.77      0.78      0.77       175
                                                     weighted avg       0.79      0.77      0.78       175


Probabilities (%) for experimental data

Dual Site Adsorption, Naphtha and Water:                              0.00
Same Site Adsorption, Naphtha and Water:                              0.00
Same Site Dissociative Adsorption, Naphtha and Water:                 100.00
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal:    0.00
Single Adsorption, Water:                                             0.00
Single Dissociative Adsorption, Water:                                0.00
Single Dissociative Adsorption, Water, Eley-Rideal:                   0.00
