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

 2022-07-18 12:02:42 
```
              precision    recall  f1-score   support

           0       0.12      0.45      0.19       121
           1       0.13      0.59      0.21       107
           2       0.97      0.63      0.77      3451
           3       0.43      0.73      0.54       103
           4       0.17      0.56      0.26       180

    accuracy                           0.63      3962
   macro avg       0.36      0.59      0.39      3962
weighted avg       0.87      0.63      0.70      3962
```

 2022-07-18 12:57:12 
```
              precision    recall  f1-score   support

           0       0.11      0.45      0.18       110
           1       0.39      0.45      0.42       420
           2       0.88      0.71      0.78      2826
           3       0.43      0.77      0.55        99
           4       0.40      0.48      0.43       507

    accuracy                           0.64      3962
   macro avg       0.44      0.57      0.47      3962
weighted avg       0.74      0.64      0.68      3962
```

 2022-07-18 13:51:46 
```
              precision    recall  f1-score   support

           0       0.11      0.40      0.18       125
           1       0.38      0.47      0.42       394
           2       0.90      0.69      0.78      2951
           3       0.52      0.67      0.58       136
           4       0.29      0.51      0.37       356

    accuracy                           0.64      3962
   macro avg       0.44      0.55      0.47      3962
weighted avg       0.76      0.64      0.68      3962
```

 2022-07-18 14:46:22 
```
              precision    recall  f1-score   support

           0       0.18      0.29      0.22       262
           1       0.42      0.43      0.42       462
           2       0.84      0.73      0.78      2600
           3       0.56      0.65      0.60       153
           4       0.37      0.47      0.42       485

    accuracy                           0.63      3962
   macro avg       0.47      0.52      0.49      3962
weighted avg       0.68      0.63      0.65      3962
```
