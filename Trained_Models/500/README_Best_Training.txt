Best fitting hyperparameters

activation1:        relu
activation2:        relu
batch_size:         64
dropout1:           0.1
dropout2:           0.1
epochs:             500
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             50
units2:             100

Accuracy tests 

Training phase:	 0.842
Testing phase:	 0.749

Confusion matrix of the testing phase (labels in alphabetical order like in the report below):

[[ 57  41   0   0   0   1   0   0]
 [ 24  61   0   0   0   0   0   0]
 [  0   0  34   0   2  13  58   0]
 [  0   0   0 119   1   0   0   0]
 [  0   0   0   0  92  10   0   0]
 [  0   1   0   0   4  89   3   1]
 [  0   0  29   0   0  11  56   0]
 [  1   0   0   1   0   0   0  91]]

Classification report (testing phase)

                                                      precision    recall  f1-score   support

                                     ER, associative       0.70      0.58      0.63        99
                                    ER, dissociative       0.59      0.72      0.65        85
LH, dissociative (HC) and molecular (H2O), same site       0.54      0.32      0.40       107
         LH, dissociative adsorption, different site       0.99      0.99      0.99       120
              LH, dissociative adsorption, same site       0.93      0.90      0.92       102
            LH, molecular adsorption, different site       0.72      0.91      0.80        98
                 LH, molecular adsorption, same site       0.48      0.58      0.53        96
                                           Power Law       0.99      0.98      0.98        93

                                            accuracy                           0.75       800
                                           macro avg       0.74      0.75      0.74       800
                                        weighted avg       0.75      0.75      0.74       800


Probabilities (%) for experimental data

ER, associative:                                                      0.00
ER, dissociative:                                                     0.00
LH, dissociative (HC) and molecular (H2O), same site:                 0.00
LH, dissociative adsorption, different site:                          100.00
LH, dissociative adsorption, same site:                               0.00
LH, molecular adsorption, different site:                             0.00
LH, molecular adsorption, same site:                                  0.00
Power Law:                                                            0.00
