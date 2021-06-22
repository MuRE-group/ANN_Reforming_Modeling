Best fitting hyperparameters

activation1:        relu
activation2:        relu
batch_size:         256
dropout1:           0.1
dropout2:           0.2
epochs:             500
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             500
units2:             100

Accuracy tests 

Training phase:	 0.846
Testing phase:	 0.800

Confusion matrix of the testing phase (labels in alphabetical order like in the report below):

[[129  67   0   0   0   2   0   0]
 [ 57 121   0   0   0   1   0   0]
 [  0   0 118   0   0   7  64   0]
 [  0   0   0 194   1   0   0   1]
 [  0   0   0   0 214   8   0   0]
 [  0   0   1   0  12 188  12   0]
 [  0   0  55   0   3  24 118   0]
 [  2   0   0   3   0   0   0 198]]

Classification report (testing phase)

                                                      precision    recall  f1-score   support

                                     ER, associative       0.69      0.65      0.67       198
                                    ER, dissociative       0.64      0.68      0.66       179
LH, dissociative (HC) and molecular (H2O), same site       0.68      0.62      0.65       189
         LH, dissociative adsorption, different site       0.98      0.99      0.99       196
              LH, dissociative adsorption, same site       0.93      0.96      0.95       222
            LH, molecular adsorption, different site       0.82      0.88      0.85       213
                 LH, molecular adsorption, same site       0.61      0.59      0.60       200
                                           Power Law       0.99      0.98      0.99       203

                                            accuracy                           0.80      1600
                                           macro avg       0.79      0.79      0.79      1600
                                        weighted avg       0.80      0.80      0.80      1600


Probabilities (%) for experimental data

ER, associative:                                                      0.00
ER, dissociative:                                                     0.00
LH, dissociative (HC) and molecular (H2O), same site:                 0.00
LH, dissociative adsorption, different site:                          0.00
LH, dissociative adsorption, same site:                               100.00
LH, molecular adsorption, different site:                             0.00
LH, molecular adsorption, same site:                                  0.00
Power Law:                                                            0.00
