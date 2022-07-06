import torch
import pandas as pd


class Tester:
    # Fire 2022
    fire_2022_tam_train = "../data/tam_sentiment_train.tsv"
    fire_2022_tam_val = "../data/tam_sentiment_dev.tsv"
     
    fire_2022_kan_train = "../data/kan_sentiment_train.tsv"
    fire_2022_kan_val = "../data/kan_sentiment_dev.tsv"
     
    fire_2022_mal_train = "../data/Mal_sentiment_train.tsv"
    fire_2022_mal_val = "../data/Mal_sentiment_dev.tsv"
    
    # Fire 2021 TODO: check if this is test set or dev
    fire_2021_tam_train = "../data/fire_2021/tamil_train.tsv"
    fire_2021_tam_val = "../data/fire_2021/tamil_test.tsv"

    fire_2021_mal_train = "../data/fire_2021/malayalam_train.tsv"
    fire_2021_mal_val = "../data/fire_2021/malayalam_test.tsv"

    # Fire 2020         
    fire_2020_tam_train = "../data/fire_2020/tamil_train.tsv"
    fire_2020_tam_val = "../data/fire_2020/tamil_dev.tsv"
    fire_2020_tam_test = "../data/fire_2020/tamil_test.tsv"

    fire_2020_mal_train = "../data/fire_2020/malayalam_train.tsv"
    fire_2020_mal_val = "../data/fire_2020/malayalam_dev.tsv"
    fire_2020_mal_test = "../data/fire_2020/malayalam_test.tsv"

    # TODO: A function that takes a model and returns a fire_2022 validation result
    # TODO: A function that takes a model and returns a fire_2021 validation result
    # TODO: A function that takes a model and returns a fire_2020 result

    # Info function about the datasets

    #

    def read_dataset(path):
        df = pd.read_csv('../data/' + path, '\t')
        texts = df.text.values
        label_cats = df.category.astype('category').cat
        label_names = label_cats.categories
        labels = label_cats.codes

        print("Texts:", len(texts))
        print("Label names:", label_names)
        return labels, texts

    def tokenize_input(texts, tokenizer):
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                                text,            
                                add_special_tokens = True,
                                max_length = 512,
                                padding = 'max_length',
                                return_attention_mask = True,
                                truncation=True,
                                return_tensors = 'pt')
        
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        return input_ids, attention_masks    
