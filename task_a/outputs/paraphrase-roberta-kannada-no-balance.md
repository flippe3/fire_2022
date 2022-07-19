## Info
This was trained on:
- [../data/kan_sentiment_train.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/kan_sentiment_train.tsv)

validated on:
 - [../data/kan_sentiment_dev.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/kan_sentiment_dev.tsv)

Model: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

 Tokenizer: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

Hyperparameters:
- Learning Rate: 3e-05
- Epochs: 4
- Batch Size: 24

 2022-07-18 18:02:35 
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         0
           1       0.71      0.45      0.55       216
           2       0.59      0.72      0.64       262
           3       0.90      0.49      0.63       204
           4       0.06      0.44      0.10         9

    accuracy                           0.56       691
   macro avg       0.45      0.42      0.39       691
weighted avg       0.71      0.56      0.60       691
```

 2022-07-18 18:06:29 
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         0
           1       0.61      0.58      0.59       147
           2       0.76      0.70      0.73       347
           3       0.77      0.60      0.68       141
           4       0.42      0.52      0.46        56

    accuracy                           0.64       691
   macro avg       0.51      0.48      0.49       691
weighted avg       0.70      0.64      0.67       691
```

 2022-07-18 18:10:27 
```
              precision    recall  f1-score   support

           0       0.02      0.11      0.03         9
           1       0.60      0.67      0.64       125
           2       0.86      0.72      0.78       386
           3       0.73      0.65      0.69       123
           4       0.39      0.56      0.46        48

    accuracy                           0.68       691
   macro avg       0.52      0.54      0.52       691
weighted avg       0.75      0.68      0.71       691
```

 2022-07-18 18:14:24 
```
              precision    recall  f1-score   support

           0       0.04      0.17      0.06        12
           1       0.64      0.61      0.63       145
           2       0.83      0.74      0.78       361
           3       0.66      0.68      0.67       108
           4       0.48      0.51      0.49        65

    accuracy                           0.67       691
   macro avg       0.53      0.54      0.53       691
weighted avg       0.72      0.67      0.69       691
```
