## Info
This was trained on [../data/tam_sentiment_train.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/tam_sentiment_train.tsv)[../data/fire_2021/tamil_train.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/fire_2021/tamil_train.tsv) and validated on [../data/tam_sentiment_dev.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/tam_sentiment_dev.tsv)

Model: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

 Tokenizer: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

Hyperparameters:
- Learning Rate: 3e-05
- Epochs: 4
- Batch Size: 16

 2022-07-11 12:16:16 
```
              precision    recall  f1-score   support

           0       0.15      0.60      0.23       107
           1       0.68      0.54      0.60       607
           2       0.91      0.77      0.84      2682
           3       0.73      0.76      0.74       169
           4       0.46      0.70      0.55       397

    accuracy                           0.72      3962
   macro avg       0.58      0.67      0.59      3962
weighted avg       0.80      0.72      0.75      3962
```

 2022-07-11 13:15:36 
```
              precision    recall  f1-score   support

           0       0.39      0.76      0.52       225
           1       0.77      0.79      0.78       470
           2       0.97      0.84      0.90      2632
           3       0.78      0.89      0.83       155
           4       0.67      0.85      0.75       480

    accuracy                           0.83      3962
   macro avg       0.72      0.82      0.76      3962
weighted avg       0.87      0.83      0.84      3962
```

 2022-07-11 14:02:16 
```
              precision    recall  f1-score   support

           0       0.80      0.81      0.80       431
           1       0.89      0.90      0.90       474
           2       0.97      0.95      0.96      2308
           3       0.92      0.93      0.92       175
           4       0.88      0.93      0.90       574

    accuracy                           0.92      3962
   macro avg       0.89      0.90      0.90      3962
weighted avg       0.92      0.92      0.92      3962
```

 2022-07-11 14:43:07 
```
              precision    recall  f1-score   support

           0       0.85      0.93      0.89       402
           1       0.93      0.94      0.93       474
           2       0.99      0.97      0.98      2308
           3       0.96      0.93      0.95       181
           4       0.93      0.95      0.94       597

    accuracy                           0.95      3962
   macro avg       0.93      0.94      0.94      3962
weighted avg       0.96      0.95      0.96      3962
```
