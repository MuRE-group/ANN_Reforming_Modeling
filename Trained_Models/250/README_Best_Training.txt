Best fitting hyperparameters

activation1:        relu
activation2:        relu
batch_size:         32
dropout1:           0.1
dropout2:           0.15
epochs:             500
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             50
units2:             50

Accuracy tests 

Training phase:	 0.769
Testing phase:	 0.652

Confusion matrix of the testing phase (labels in alphabetical order like in the report below):

[[18 27  0  0  0  0  0  0]
 [18 19  0  0  0  0  0  0]
 [ 0  1  8  0  3  8 38  0]
 [ 0  0  0 47  0  0  0  0]
 [ 1  1  1  0 44  3  0  0]
 [ 4  2  2  0  4 45  1  0]
 [ 0  0 11  0  3  7 29  0]
 [ 2  1  0  0  0  0  1 51]]

Classification report (testing phase)

                                                      precision    recall  f1-score   support

                                     ER, associative       0.42      0.40      0.41        45
                                    ER, dissociative       0.37      0.51      0.43        37
LH, dissociative (HC) and molecular (H2O), same site       0.36      0.14      0.20        58
         LH, dissociative adsorption, different site       1.00      1.00      1.00        47
              LH, dissociative adsorption, same site       0.81      0.88      0.85        50
            LH, molecular adsorption, different site       0.71      0.78      0.74        58
                 LH, molecular adsorption, same site       0.42      0.58      0.49        50
                                           Power Law       1.00      0.93      0.96        55

                                            accuracy                           0.65       400
                                           macro avg       0.64      0.65      0.64       400
                                        weighted avg       0.65      0.65      0.64       400


Probabilities (%) for experimental data

ER, associative:                                                      0.00
ER, dissociative:                                                     0.00
LH, dissociative (HC) and molecular (H2O), same site:                 0.00
LH, dissociative adsorption, different site:                          100.00
LH, dissociative adsorption, same site:                               0.00
LH, molecular adsorption, different site:                             0.00
LH, molecular adsorption, same site:                                  0.00
Power Law:                                                            0.00
