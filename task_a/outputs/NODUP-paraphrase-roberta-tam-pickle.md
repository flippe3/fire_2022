## Info
This was trained on:
- [../data/new_tam_train.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/new_tam_train.tsv)

validated on:
 - [../data/tam_sentiment_dev.tsv](https://github.com/flippe3/fire_2022/tree/master/task_a/data/../data/tam_sentiment_dev.tsv)

Model: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

 Tokenizer: [sentence-transformers/paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)

Hyperparameters:
- Learning Rate: 3e-05
- Epochs: 4
- Batch Size: 24

 2022-07-27 00:31:50 
```
              precision    recall  f1-score   support

           0       0.11      0.42      0.17       110
           1       0.39      0.45      0.42       417
           2       0.89      0.70      0.79      2871
           3       0.51      0.73      0.60       123
           4       0.35      0.49      0.41       441

    accuracy                           0.65      3962
   macro avg       0.45      0.56      0.48      3962
weighted avg       0.75      0.65      0.68      3962
```

 2022-07-27 00:51:28 
```
              precision    recall  f1-score   support

           0       0.18      0.43      0.26       186
           1       0.34      0.52      0.41       319
           2       0.88      0.72      0.79      2738
           3       0.58      0.66      0.62       154
           4       0.44      0.47      0.46       565

    accuracy                           0.66      3962
   macro avg       0.48      0.56      0.51      3962
weighted avg       0.73      0.66      0.68      3962
```

 2022-07-27 01:11:09 
```
              precision    recall  f1-score   support

           0       0.20      0.37      0.26       237
           1       0.41      0.47      0.43       418
           2       0.83      0.74      0.78      2552
           3       0.57      0.64      0.60       159
           4       0.46      0.47      0.46       596

    accuracy                           0.64      3962
   macro avg       0.49      0.54      0.51      3962
weighted avg       0.68      0.64      0.66      3962
```

 2022-07-27 01:30:50 
```
              precision    recall  f1-score   support

           0       0.20      0.30      0.24       293
           1       0.44      0.44      0.44       479
           2       0.81      0.74      0.78      2469
           3       0.56      0.64      0.60       155
           4       0.43      0.46      0.45       566

    accuracy                           0.63      3962
   macro avg       0.49      0.52      0.50      3962
weighted avg       0.66      0.63      0.64      3962
```
