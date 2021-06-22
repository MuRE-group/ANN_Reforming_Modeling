Best fitting hyperparameters

activation1:        relu
activation2:        relu
batch_size:         256
dropout1:           0.2
dropout2:           0.15
epochs:             500
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             100
units2:             50

Accuracy tests 

Training phase:	 0.801
Testing phase:	 0.780

Confusion matrix of the testing phase (labels in alphabetical order like in the report below):

[[ 90  49   0   0   0   0   0   0]
 [ 57  96   0   0   0   2   0   0]
 [  0   0  75   0   0   5  66   1]
 [  0   0   0 145   0   0   0  14]
 [  0   0   3   0 137   8   2   1]
 [  0   0   2   0   0 150   1   1]
 [  0   0  29   0   0  18 114   0]
 [  1   2   0   0   0   0   2 129]]

Classification report (testing phase)

                                                      precision    recall  f1-score   support

                                     ER, associative       0.61      0.65      0.63       139
                                    ER, dissociative       0.65      0.62      0.64       155
LH, dissociative (HC) and molecular (H2O), same site       0.69      0.51      0.59       147
         LH, dissociative adsorption, different site       1.00      0.91      0.95       159
              LH, dissociative adsorption, same site       1.00      0.91      0.95       151
            LH, molecular adsorption, different site       0.82      0.97      0.89       154
                 LH, molecular adsorption, same site       0.62      0.71      0.66       161
                                           Power Law       0.88      0.96      0.92       134

                                            accuracy                           0.78      1200
                                           macro avg       0.78      0.78      0.78      1200
                                        weighted avg       0.78      0.78      0.78      1200


Probabilities (%) for experimental data

ER, associative:                                                      0.00
ER, dissociative:                                                     0.00
LH, dissociative (HC) and molecular (H2O), same site:                 0.00
LH, dissociative adsorption, different site:                          0.00
LH, dissociative adsorption, same site:                               0.00
LH, molecular adsorption, different site:                             0.00
LH, molecular adsorption, same site:                                  0.00
Power Law:                                                            100.00
