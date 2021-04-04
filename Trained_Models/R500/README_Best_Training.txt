Best fitting hyperparameters

activation1:        elu
batch_size:         512
dropout1:           0.15
epochs:             5000
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             100

Accuracy tests 

Training phase:	 0.917
Testing phase:	 0.871

Confusion matrix

[[ 72   7   1   7  19   1   1]
 [  7  88   1   1   0   0   1]
 [  0   0 113   0   0   2   0]
 [  0   3   0  89   4   0   0]
 [  5   1   0   6  68   1   7]
 [  0   0   5   0   0  84   1]
 [  0   0   0   7   2   0  96]]

Classification report

                                                                   precision    recall  f1-score   support

                          Dual Site Adsorption, Naphtha and Water       0.86      0.67      0.75       108
                          Same Site Adsorption, Naphtha and Water       0.89      0.90      0.89        98
             Same Site Dissociative Adsorption, Naphtha and Water       0.94      0.98      0.96       115
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal       0.81      0.93      0.86        96
                                         Single Adsorption, Water       0.73      0.77      0.75        88
                            Single Dissociative Adsorption, Water       0.95      0.93      0.94        90
               Single Dissociative Adsorption, Water, Eley-Rideal       0.91      0.91      0.91       105

                                                         accuracy                           0.87       700
                                                        macro avg       0.87      0.87      0.87       700
                                                     weighted avg       0.87      0.87      0.87       700


Probabilities (%) for experimental data

Dual Site Adsorption, Naphtha and Water:                              0.00
Same Site Adsorption, Naphtha and Water:                              0.00
Same Site Dissociative Adsorption, Naphtha and Water:                 100.00
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal:    0.00
Single Adsorption, Water:                                             0.00
Single Dissociative Adsorption, Water:                                0.00
Single Dissociative Adsorption, Water, Eley-Rideal:                   0.00
