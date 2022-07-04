# [Findings fire 2021](https://arxiv.org/pdf/2111.09811.pdf)

## Tamil top 3 (in order):
- [CIA_NITT](http://ceur-ws.org/Vol-3159/T6-22.pdf) (F1: 0.71): The authors proposed a system that uses a pretrained XLM-RoBERTa for sequence classification. They tokenize the input text using the SentencePiece tokenizer, which is then fed as embeddings to be fine-tuned for the XLM-RoBERTa model
- [ZYBank-AI](http://ceur-ws.org/Vol-3159/T6-5.pdf) (F1: 0.68): The authors based their experiments on the XLM-RoBERTa as well. To improve the results, they have added self-attention to the 12 hidden layers of the XLMRoBERTA. Furthermore, they propose a two-stage pipeline for the task at hand. In the first stage, the model is trained on data from Dravidian-CodeMix-FIRE 2020. In the second stage, the pre-trained model is fine-tuned on the Dravidian-CodeMix-FIRE 2021 and evaluated on test data.
- [IIITT-Pawan](http://ceur-ws.org/Vol-3159/T6-6.pdf) (F1: 0.63): The authors proposed an ensemble of several fine-tuned language models for sequence classification: BERT, MuRIL, XLM-RoBERTa, DistilBERT. Each of the classifiers is separately trained on training data. During testing, soft voting is employed among all of these classifiers to predict the most likely class.

## Malayalam top 3 (in order):
- [ZYBank-AI](http://ceur-ws.org/Vol-3159/T6-5.pdf) (F1: 0.80)
- [CIA_NITT](http://ceur-ws.org/Vol-3159/T6-22.pdf) (F1: 0.75)
- [SOA_NLP](http://ceur-ws.org/Vol-3159/T6-8.pdf) (F1: 0.73): The authors proposed the following two ensemble models for tackling the problem at hand: an ensemble of support vector machine, logistic regression and random forest for Kannada-English texts and an ensemble of support vector machine and logistic regression for Malayalam-English and Tamil-English texts. 

## Kannada top 3 (in order):
- [SSNCSE_NLP](http://ceur-ws.org/Vol-3159/T6-18.pdf) (F1: 0.63): The authors employed a variety of feature extraction techniques and concluded that the count, TF-IDF based vectorization, and multilingual transformer encoding technique performs well on the code-mix polarity labelling task. With these features, and acheived a weighted F1 score of 0.588 for the Tamil-English task, 0.69 for the Malayalam-English task and 0.63 for the Kannada-English tasks respectively.
- [MUCIC](http://ceur-ws.org/Vol-3159/T6-2.pdf) (F1: 0.63): The authors extracted the character level and syllable level features from the text, which were then used to create the TF-IDF feature vectors. The authors have documented three models, namely: a logistic regression model, an LSTM classifier, and a multilayer perceptron classifier, to classify the messages. The TF-IDF feature vectors are fed to these models, which in turn are trained on the classification task.
- [CIA_NITT](http://ceur-ws.org/Vol-3159/T6-22.pdf) (F1: 0.63)

[Other papers](http://ceur-ws.org/Vol-3159/)

[CIA_NITT could also be](http://ceur-ws.org/Vol-2826/T4-14.pdf)

[Fire 2021 dataset analysis](https://dl.acm.org/doi/pdf/10.1145/3503162.3503177)