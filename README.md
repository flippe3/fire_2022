# Fire 2022

Official repo for team EISLAB in [Sentiment Analysis and Homophobia detection  of YouTube comments in Code-Mixed Dravidian Languages
](https://sites.google.com/view/dravidiancodemix-2022/home?authuser=0)

## Ideas
- Write simple pipeline to test model on validation, fire2020, fire2021
- Try other models form [SBERT](https://www.sbert.net/docs/pretrained_models.html)
- Implement K-fold
- Try zero-shot [meta learning](http://learn2learn.net/) 
- Compare fire2021 and fire2022
- Run [lang_detect](https://pypi.org/project/langdetect/) and store outputs

## Questions 
- Can different language can be leveraged?
    - Translations?
- Leverage commonality between the tasks?
    - Evaluate the classifier on both tasks and see if there is any transference
- What can be done with imbalancing?
    - K-folds
    - Meta-learning
- Can any meaningful pre-training be done?
- What pre-processing should be done?

## Models to run
- [Pharaphrase XLM-Roberta](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1)
- [XLM-Roberta](https://huggingface.co/xlm-roberta-large)
- [Multilanguage BERT](bert-base-multilingual-cased)
- [BERT](https://huggingface.co/bert-base-uncased)

## Tested

## Validation Results (Fire 2022)

## Fire 2021 Results

## Fire 2020 Results
