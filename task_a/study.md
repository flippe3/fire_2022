# Study Task A

## Ideas
- ~~Write simple pipeline to test model on validation, fire2020, fire2021~~
- Try other models form [SBERT](https://www.sbert.net/docs/pretrained_models.html)
- Implement K-fold
- Try zero-shot [meta learning](http://learn2learn.net/) 
- Compare fire2021 and fire2022
- Run [lang_detect](https://pypi.org/project/langdetect/) and store outputs
- Look at [imbalanced learn](https://imbalanced-learn.org/stable/)

## [Duplicate Findings (shown in dataset_duplicates.ipynb)](https://github.com/flippe3/fire_2022/blob/master/task_a/code/dataset_duplicates.ipynb)
- There are duplicate examples in task_a training data.
- There are duplicate examples in task_b validation data.
- The entire fire 2022 validation set is part of fire2021 training data.

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
- [XLM-roberta-base](https://huggingface.co/xlm-roberta-base)
## Validation Results (Fire 2022)
- [XLM-roberta-base](https://github.com/flippe3/fire_2022/blob/master/task_a/outputs/xlm-roberta-base-tamil)
## Fire 2021 Results

## Fire 2020 Results

