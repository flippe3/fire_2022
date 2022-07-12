## Info
This was trained on:
- [../data/tam_sentiment_train.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/tam_sentiment_train.tsv)
- [../data/fire_2020/tamil_train.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/fire_2020/tamil_train.tsv)

validated on:
 - [../data/tam_sentiment_dev.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/tam_sentiment_dev.tsv)

Model: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

 Tokenizer: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

Hyperparameters:
- Learning Rate: 3e-05
- Epochs: 4
- Batch Size: 24

 2022-07-11 16:07:42 
```
              precision    recall  f1-score   support

           0       0.15      0.44      0.22       147
           1       0.41      0.51      0.46       387
           2       0.90      0.71      0.80      2872
           3       0.62      0.62      0.62       177
           4       0.32      0.51      0.39       379

    accuracy                           0.66      3962
   macro avg       0.48      0.56      0.50      3962
weighted avg       0.76      0.66      0.69      3962
```

 2022-07-11 16:33:01 
```
              precision    recall  f1-score   support

           0       0.20      0.42      0.27       204
           1       0.57      0.52      0.55       525
           2       0.87      0.76      0.81      2561
           3       0.66      0.59      0.62       199
           4       0.41      0.53      0.46       473

    accuracy                           0.68      3962
   macro avg       0.54      0.56      0.54      3962
weighted avg       0.73      0.68      0.70      3962
```

 2022-07-11 16:58:20 
```
              precision    recall  f1-score   support

           0       0.32      0.44      0.37       320
           1       0.52      0.59      0.55       424
           2       0.85      0.79      0.82      2454
           3       0.60      0.69      0.64       153
           4       0.52      0.52      0.52       611

    accuracy                           0.69      3962
   macro avg       0.56      0.60      0.58      3962
weighted avg       0.71      0.69      0.70      3962
```

 2022-07-11 17:30:58 
```
              precision    recall  f1-score   support

           0       0.37      0.41      0.39       391
           1       0.56      0.56      0.56       481
           2       0.85      0.79      0.82      2411
           3       0.61      0.68      0.64       157
           4       0.46      0.53      0.49       522

    accuracy                           0.69      3962
   macro avg       0.57      0.60      0.58      3962
weighted avg       0.71      0.69      0.70      3962
```
