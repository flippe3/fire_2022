import pandas as pd
import torch

class MTL_Dataset:
    def __init__(self):
        self.kan_sentiment_t = "../task_a/data/new_kan_train.tsv"
        self.mal_sentiment_t = "../task_a/data/new_mal_train.tsv"
        self.tam_sentiment_t = "../task_a/data/new_tam_train.tsv"

        self.kan_sentiment_d = "../task_a/data/kan_sentiment_dev.tsv"
        self.mal_sentiment_d = "../task_a/data/Mal_sentiment_dev.tsv"
        self.tam_sentiment_d = "../task_a/data/tam_sentiment_dev.tsv"

        self.eng_phobia_t = "../task_b/data/eng_3_train.tsv" 
        self.mal_phobia_t = "../task_b/data/new_mal_train.tsv" 
        self.tam_phobia_t = "../task_b/data/new_tam_train.tsv" 
        self.eng_tam_phobia_t = "../task_b/data/new_eng_tam_train.tsv" 

        self.eng_phobia_d = "../task_b/data/eng_3_dev.tsv" 
        self.mal_phobia_d = "../task_b/data/mal_3_dev.csv" 
        self.tam_phobia_d = "../task_b/data/tam_3_dev.tsv" 
        self.eng_tam_phobia_d = "../task_b/data/eng-tam_3_dev.tsv" 
    
    def get_dataset_dict(self):
        dataset_dict = {
            "kan_sentiment": {"train": pd.read_csv(self.kan_sentiment_t, '\t'), "val": pd.read_csv(self.kan_sentiment_d, '\t')},
            "mal_sentiment": {"train": pd.read_csv(self.mal_sentiment_t, '\t'), "val": pd.read_csv(self.mal_sentiment_d, '\t')},
            "tam_sentiment": {"train": pd.read_csv(self.tam_sentiment_t, '\t'), "val": pd.read_csv(self.tam_sentiment_d, '\t')},
            
            "eng_phobia": {"train": pd.read_csv(self.eng_phobia_t, '\t'), "val": pd.read_csv(self.eng_phobia_d, '\t')},
            "mal_phobia": {"train": pd.read_csv(self.mal_phobia_t, '\t'), "val": pd.read_csv(self.mal_phobia_d, '\t')},
            "tam_phobia": {"train": pd.read_csv(self.tam_phobia_t, '\t'), "val": pd.read_csv(self.tam_phobia_d, '\t')},
            "eng_tam_phobia": {"train": pd.read_csv(self.eng_tam_phobia_t, '\t'), "val": pd.read_csv(self.eng_tam_phobia_d, '\t')}
        }
        
        return dataset_dict

    def read_dataset(self, path, test=False):
        if path[-4:] == ".tsv":
            df = pd.read_csv(path, '\t')
        else:
            df = pd.read_csv(path)
        
        df = df.dropna(subset=['text','category'])
        df = df.drop_duplicates(subset='text')

        texts = df.text.values
        
        if test == False:
            label_cats = df.category.astype('category').cat
            label_names = label_cats.categories
            labels = label_cats.codes

            return labels, texts
    
    def tokenize_input(self, texts, tokenizer):        
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
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks 
