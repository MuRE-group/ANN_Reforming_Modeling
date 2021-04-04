Best fitting hyperparameters

activation1:        relu
batch_size:         128
dropout1:           0.4
epochs:             5000
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             100

Accuracy tests 

Training phase:	 1.000
Testing phase:	 0.343

Confusion matrix

[[2 2 1 2 0 1 1]
 [0 8 0 1 1 0 1]
 [0 2 3 0 0 4 0]
 [2 2 0 3 1 2 0]
 [2 1 0 2 1 1 3]
 [0 0 2 0 1 5 0]
 [2 0 0 5 1 3 2]]

Classification report

                                                                   precision    recall  f1-score   support

                          Dual Site Adsorption, Naphtha and Water       0.25      0.22      0.24         9
                          Same Site Adsorption, Naphtha and Water       0.53      0.73      0.62        11
             Same Site Dissociative Adsorption, Naphtha and Water       0.50      0.33      0.40         9
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal       0.23      0.30      0.26        10
                                         Single Adsorption, Water       0.20      0.10      0.13        10
                            Single Dissociative Adsorption, Water       0.31      0.62      0.42         8
               Single Dissociative Adsorption, Water, Eley-Rideal       0.29      0.15      0.20        13

                                                         accuracy                           0.34        70
                                                        macro avg       0.33      0.35      0.32        70
                                                     weighted avg       0.33      0.34      0.32        70


Probabilities (%) for experimental data

Dual Site Adsorption, Naphtha and Water:                              0.00
Same Site Adsorption, Naphtha and Water:                              0.00
Same Site Dissociative Adsorption, Naphtha and Water:                 0.00
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal:    0.00
Single Adsorption, Water:                                             0.00
Single Dissociative Adsorption, Water:                                100.00
Single Dissociative Adsorption, Water, Eley-Rideal:                   0.00
