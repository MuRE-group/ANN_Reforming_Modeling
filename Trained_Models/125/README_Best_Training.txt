Best fitting hyperparameters

activation1:        relu
activation2:        relu
batch_size:         256
dropout1:           0.15
dropout2:           0.2
epochs:             500
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             50
units2:             50

Accuracy tests 

Training phase:	 0.710
Testing phase:	 0.595

Confusion matrix of the testing phase (labels in alphabetical order like in the report below):

[[10 11  0  1  0  0  0  0]
 [14 10  0  0  0  0  0  0]
 [ 0  0  4  0  2  7 14  0]
 [ 0  0  0 25  0  0  0  0]
 [ 0  0  2  0 23  4  0  0]
 [ 0  0  1  0  1 18  1  0]
 [ 0  0 10  0  2  5  9  0]
 [ 1  1  0  3  0  1  0 20]]

Classification report (testing phase)

                                                      precision    recall  f1-score   support

                                     ER, associative       0.40      0.45      0.43        22
                                    ER, dissociative       0.45      0.42      0.43        24
LH, dissociative (HC) and molecular (H2O), same site       0.24      0.15      0.18        27
         LH, dissociative adsorption, different site       0.86      1.00      0.93        25
              LH, dissociative adsorption, same site       0.82      0.79      0.81        29
            LH, molecular adsorption, different site       0.51      0.86      0.64        21
                 LH, molecular adsorption, same site       0.38      0.35      0.36        26
                                           Power Law       1.00      0.77      0.87        26

                                            accuracy                           0.59       200
                                           macro avg       0.58      0.60      0.58       200
                                        weighted avg       0.59      0.59      0.58       200


Probabilities (%) for experimental data

ER, associative:                                                      0.00
ER, dissociative:                                                     0.00
LH, dissociative (HC) and molecular (H2O), same site:                 0.00
LH, dissociative adsorption, different site:                          0.00
LH, dissociative adsorption, same site:                               0.00
LH, molecular adsorption, different site:                             0.00
LH, molecular adsorption, same site:                                  0.00
Power Law:                                                            100.00
