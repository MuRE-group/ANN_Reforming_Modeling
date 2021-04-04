Best fitting hyperparameters

activation1:        elu
batch_size:         128
dropout1:           0.15
epochs:             5000
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             100

Accuracy tests 

Training phase:	 0.921
Testing phase:	 0.851

Confusion matrix

[[30  2  0  1 11  1  4]
 [ 7 32  1  1  0  0  0]
 [ 0  0 50  0  0  0  0]
 [ 0  1  0 53  3  0  4]
 [ 3  0  0  1 36  1  3]
 [ 0  0  0  0  1 43  1]
 [ 0  0  0  2  4  0 54]]

Classification report

                                                                   precision    recall  f1-score   support

                          Dual Site Adsorption, Naphtha and Water       0.75      0.61      0.67        49
                          Same Site Adsorption, Naphtha and Water       0.91      0.78      0.84        41
             Same Site Dissociative Adsorption, Naphtha and Water       0.98      1.00      0.99        50
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal       0.91      0.87      0.89        61
                                         Single Adsorption, Water       0.65      0.82      0.73        44
                            Single Dissociative Adsorption, Water       0.96      0.96      0.96        45
               Single Dissociative Adsorption, Water, Eley-Rideal       0.82      0.90      0.86        60

                                                         accuracy                           0.85       350
                                                        macro avg       0.86      0.85      0.85       350
                                                     weighted avg       0.86      0.85      0.85       350


Probabilities (%) for experimental data

Dual Site Adsorption, Naphtha and Water:                              0.00
Same Site Adsorption, Naphtha and Water:                              0.00
Same Site Dissociative Adsorption, Naphtha and Water:                 100.00
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal:    0.00
Single Adsorption, Water:                                             0.00
Single Dissociative Adsorption, Water:                                0.00
Single Dissociative Adsorption, Water, Eley-Rideal:                   0.00
