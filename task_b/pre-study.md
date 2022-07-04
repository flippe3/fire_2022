# [Findings ACL2022 Homophobia/Transphobia Detection in social media comments](https://aclanthology.org/2022.ltedi-1.57.pdf)

## Tamil top 3 (in order):
- [ARGUABLY](https://aclanthology.org/2022.ltedi-1.55/) (F1: 0.87): they we present our classification system; given comments, it predicts whether or not it contains any form of homophobia/transphobia with a zero-shot learning framework.

- [NAYEL](https://aclanthology.org/2022.ltedi-1.42.pdf) (F1: 0.84): has experimented with TF-IDF with bigram models to vectorize comments. Then they implemented a set of classification algorithms like Support Vector Machine, Random Forest, Passive Aggressive Classifier, Gaussian Naïve Bayes, and Multi-Layer Perceptron. From these models, they submitted a support vector machine as the best model because it gave high accuracy compared to other models.

- [UMUTeam](https://aclanthology.org/2022.ltedi-1.16.pdf) (F1: 0.80): This team used neural networks that combine several features sets, including linguistic components extracted from a self-developed tool and contextual and non-contextual sentence embeddings. This team got 7th, 3rd, and 2nd ranks in English, Tamil, and Tamil-English.

## English top 3 (in order):
- [Ablimet](https://aclanthology.org/2022.ltedi-1.19.pdf) (F1: 0.57): has used a fine-tuning approach to the pre-trained language model. This model processes the target data and normalizes its output by a layer normalization module, followed by two fully connected layers. The pre-trained language model they used is the Roberta-base model for the English subtask, Tamil-Roberta for Tamil, and Tamil-English subtasks.

- [Sammaan](https://aclanthology.org/2022.ltedi-1.39.pdf) (F1: 0.49): This team used an ensemble of transformer-based models to build the classifier. They got 2nd rank for English, 8th rank for Tamil, and 10th rank for Tamil-English. They experimented with models BERT, RoBERTa, HateBERT, IndicBERT, XGBoost, Random Forest classifier, and Bayesian Optimization.

- [Nozza](https://aclanthology.org/2022.ltedi-1.37.pdf) (F1: 0.45): team used fine-tuned models, and they selected two large language models, BERT and RoBERTa, to classify the task and gave the result which is shown above. Also, they chose HateBERT to provide more accuracy than other models, while this better results than the BERT model. They experimented with the ensemble modeling created with a meta-classifier that treats the predicted label of distinct machine learning classifiers as a vote towards the final label they give as a prediction. Also, they gave two frameworks for ensemble: majority voting and weighted voting.

## Tamil-English top 3 (in order):
- [ARGUABLY](https://aclanthology.org/2022.ltedi-1.55/) (F1: 0.61)

- [UMUTeam](https://aclanthology.org/2022.ltedi-1.16.pdf) (F1: 0.58)

- [bitsa_nlp](https://aclanthology.org/2022.ltedi-1.18.pdf) (F1: 0.58): has used famous distinctive models primarily based totally on the transformer architecture and a data augmentation approach for oversampling the English, Tamil, and Tamil-English datasets. They implemented various pre-trained language models based on the Transformer architectures, namely BERT, mBERT / multilingual BERT, XLM-RoBERTa, IndicBERT, and HateBERT, to classify detecting homophobic and transphobic contents. 


[ACL 2022 dataset analysis](https://arxiv.org/pdf/2109.00227.pdf)