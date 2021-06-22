Best fitting hyperparameters

activation1:        relu
activation2:        relu
batch_size:         32
dropout1:           0.2
dropout2:           0.2
epochs:             250
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             50
units2:             50

Accuracy tests 

Training phase:	 0.731
Testing phase:	 0.412

Confusion matrix of the testing phase (labels in alphabetical order like in the report below):

[[ 2  9  0  0  0  0  0  0]
 [ 5  5  0  0  0  0  0  0]
 [ 0  0  1  0  0  3  2  0]
 [ 0  0  0 11  1  0  0  0]
 [ 0  1  0  0  3  3  0  0]
 [ 0  0  2  1  1  5  0  0]
 [ 0  1  8  1  0  3  0  0]
 [ 2  2  0  0  0  2  0  6]]

Classification report (testing phase)

                                                      precision    recall  f1-score   support

                                     ER, associative       0.22      0.18      0.20        11
                                    ER, dissociative       0.28      0.50      0.36        10
LH, dissociative (HC) and molecular (H2O), same site       0.09      0.17      0.12         6
         LH, dissociative adsorption, different site       0.85      0.92      0.88        12
              LH, dissociative adsorption, same site       0.60      0.43      0.50         7
            LH, molecular adsorption, different site       0.31      0.56      0.40         9
                 LH, molecular adsorption, same site       0.00      0.00      0.00        13
                                           Power Law       1.00      0.50      0.67        12

                                            accuracy                           0.41        80
                                           macro avg       0.42      0.41      0.39        80
                                        weighted avg       0.44      0.41      0.40        80


Probabilities (%) for experimental data

ER, associative:                                                      0.00
ER, dissociative:                                                     0.00
LH, dissociative (HC) and molecular (H2O), same site:                 0.00
LH, dissociative adsorption, different site:                          0.00
LH, dissociative adsorption, same site:                               0.00
LH, molecular adsorption, different site:                             0.00
LH, molecular adsorption, same site:                                  0.00
Power Law:                                                            100.00
