Best fitting hyperparameters

activation1:        elu
batch_size:         64
dropout1:           0.15
epochs:             5000
input_dim:          954
loss:               categorical_crossentropy
optimizer:          adam
units1:             100

Accuracy tests 

Training phase:	 0.922
Testing phase:	 0.876

Confusion matrix

[[148  14   1   1  35   0   1]
 [ 18 189   1   3   3   0   1]
 [  0   2 182   1   0   3   0]
 [ 11   1   1 181   5   0   3]
 [ 23   2   0   1 166   0   4]
 [  6   0   7   0   0 186   0]
 [ 13   0   0   1  10   1 175]]

Classification report

                                                                   precision    recall  f1-score   support

                          Dual Site Adsorption, Naphtha and Water       0.68      0.74      0.71       200
                          Same Site Adsorption, Naphtha and Water       0.91      0.88      0.89       215
             Same Site Dissociative Adsorption, Naphtha and Water       0.95      0.97      0.96       188
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal       0.96      0.90      0.93       202
                                         Single Adsorption, Water       0.76      0.85      0.80       196
                            Single Dissociative Adsorption, Water       0.98      0.93      0.96       199
               Single Dissociative Adsorption, Water, Eley-Rideal       0.95      0.88      0.91       200

                                                         accuracy                           0.88      1400
                                                        macro avg       0.88      0.88      0.88      1400
                                                     weighted avg       0.88      0.88      0.88      1400


Probabilities (%) for experimental data

Dual Site Adsorption, Naphtha and Water:                              0.00
Same Site Adsorption, Naphtha and Water:                              0.00
Same Site Dissociative Adsorption, Naphtha and Water:                 100.00
Same Site Dissociative Adsorption, Naphtha and Water, Eley-Rideal:    0.00
Single Adsorption, Water:                                             0.00
Single Dissociative Adsorption, Water:                                0.00
Single Dissociative Adsorption, Water, Eley-Rideal:                   0.00
