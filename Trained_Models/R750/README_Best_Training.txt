Best fitting hyperparameters

activation1:        elu
batch_size:         512
dropout1:           0.01
epochs:             5000
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             250

Accuracy tests 

Training phase:	 0.916
Testing phase:	 0.870

Confusion matrix

[[121   6   0   1  21   0   3]
 [ 23 142   1   0   1   0   0]
 [  0   0 138   1   0   1   0]
 [  7   0   0 108   1   0   4]
 [ 22   2   0   1 108   0  15]
 [  0   0   9   0   0 143   1]
 [  9   0   2   4   1   0 154]]

Classification report

                                                                   precision    recall  f1-score   support

                          Dual Site Adsorption, Naphtha and Water       0.66      0.80      0.72       152
                          Same Site Adsorption, Naphtha and Water       0.95      0.85      0.90       167
             Same Site Dissociative Adsorption, Naphtha and Water       0.92      0.99      0.95       140
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal       0.94      0.90      0.92       120
                                         Single Adsorption, Water       0.82      0.73      0.77       148
                            Single Dissociative Adsorption, Water       0.99      0.93      0.96       153
               Single Dissociative Adsorption, Water, Eley-Rideal       0.87      0.91      0.89       170

                                                         accuracy                           0.87      1050
                                                        macro avg       0.88      0.87      0.87      1050
                                                     weighted avg       0.88      0.87      0.87      1050


Probabilities (%) for experimental data

Dual Site Adsorption, Naphtha and Water:                              0.00
Same Site Adsorption, Naphtha and Water:                              0.00
Same Site Dissociative Adsorption, Naphtha and Water:                 100.00
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal:    0.00
Single Adsorption, Water:                                             0.00
Single Dissociative Adsorption, Water:                                0.00
Single Dissociative Adsorption, Water, Eley-Rideal:                   0.00
