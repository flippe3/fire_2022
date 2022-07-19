## Info
This was trained on:
- [../data/tam_sentiment_train.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/tam_sentiment_train.tsv)

validated on:
 - [../data/tam_sentiment_dev.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/tam_sentiment_dev.tsv)

Model: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

 Tokenizer: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

Hyperparameters:
- Learning Rate: 3e-05
- Epochs: 4
- Batch Size: 24

 2022-07-18 18:47:38 
```
              precision    recall  f1-score   support

           0       0.10      0.47      0.17        94
           1       0.12      0.54      0.20       110
           2       0.93      0.66      0.77      3192
           3       0.56      0.61      0.58       160
           4       0.32      0.49      0.39       406

    accuracy                           0.63      3962
   macro avg       0.41      0.55      0.42      3962
weighted avg       0.81      0.63      0.69      3962
```

 2022-07-18 19:19:38 
```
              precision    recall  f1-score   support

           0       0.14      0.45      0.21       136
           1       0.38      0.49      0.42       368
           2       0.90      0.69      0.78      2920
           3       0.53      0.68      0.60       138
           4       0.35      0.54      0.43       400

    accuracy                           0.65      3962
   macro avg       0.46      0.57      0.49      3962
weighted avg       0.76      0.65      0.69      3962
```

 2022-07-18 19:56:20 
```
              precision    recall  f1-score   support

           0       0.21      0.32      0.25       284
           1       0.44      0.44      0.44       480
           2       0.84      0.74      0.79      2552
           3       0.56      0.64      0.60       152
           4       0.41      0.51      0.45       494

    accuracy                           0.64      3962
   macro avg       0.49      0.53      0.50      3962
weighted avg       0.68      0.64      0.66      3962
```

 2022-07-18 20:31:06 
```
              precision    recall  f1-score   support

           0       0.20      0.31      0.25       288
           1       0.45      0.45      0.45       477
           2       0.83      0.75      0.78      2506
           3       0.61      0.65      0.63       167
           4       0.42      0.49      0.45       524

    accuracy                           0.64      3962
   macro avg       0.50      0.53      0.51      3962
weighted avg       0.67      0.64      0.65      3962
```
