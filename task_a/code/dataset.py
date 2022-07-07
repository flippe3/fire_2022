from lib2to3.pgen2 import token
from sklearn.metrics import classification_report
import torch
import pandas as pd
from util import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np
from datetime import datetime

class Dataset:
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

    def get_dataset(self, tokenizer, train_file):
        labels, texts = read_dataset(train_file)
        inputs, masks =  tokenize_input(texts, tokenizer)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(inputs, masks, labels)
        return dataset

    def get_fire_2022_dataset(self, tokenizer):
        tam_train = self.get_dataset(tokenizer, self.fire_2022_tam_train)
        tam_val = self.get_dataset(tokenizer, self.fire_2022_tam_val)

        kan_train = self.get_dataset(tokenizer, self.fire_2022_kan_train)
        kan_val = self.get_dataset(tokenizer, self.fire_2022_kan_val)

        mal_train = self.get_dataset(tokenizer, self.fire_2022_mal_train)
        mal_val = self.get_dataset(tokenizer, self.fire_2022_mal_val)

        return tam_train, tam_val, kan_train, kan_val, mal_train, mal_val
         
    def get_fire_2021_dataset(self, tokenizer):
        tam_train = self.get_dataset(tokenizer, self.fire_2021_tam_train)
        tam_val = self.get_dataset(tokenizer, self.fire_2021_tam_val)

        mal_train = self.get_dataset(tokenizer, self.fire_2021_mal_train)
        mal_val = self.get_dataset(tokenizer, self.fire_2021_mal_val)

        return tam_train, tam_val, mal_train, mal_val
    
    def get_fire_2020_dataset(self, tokenizer):
        tam_train = self.get_dataset(tokenizer, self.fire_2020_tam_train)
        tam_val = self.get_dataset(tokenizer, self.fire_2020_tam_val)

        mal_train = self.get_dataset(tokenizer, self.fire_2020_mal_train)
        mal_val = self.get_dataset(tokenizer, self.fire_2020_mal_val)

        return tam_train, tam_val, mal_train, mal_val


    def fire_validation(self, model, tokenizer, device, output_file, year=2022, BS=16, dataset='tam'):
        if year == 2022:
            _, tam_val, _, kan_val, _, mal_val = self.get_fire_2022_dataset(tokenizer)
        elif year == 2021:
           _, tam_val, _, mal_val = self.get_fire_2021_dataset(tokenizer)
        elif year == 2020:
           _, tam_val, _, mal_val = self.get_fire_2020_dataset(tokenizer)


        if dataset == 'tam':
            loader = DataLoader(tam_val, sampler = SequentialSampler(tam_val), batch_size=BS)
        elif dataset == 'kan' and year == 2022:
            loader = DataLoader(kan_val, sampler = SequentialSampler(tam_val), batch_size=BS) 
        elif dataset == 'mal':
            loader = DataLoader(mal_val, sampler = SequentialSampler(tam_val), batch_size=BS) 

        print(f"{dataset} validation: {loader * BS}")
        
        vbar = tqdm(enumerate(loader), total=len(loader), desc= dataset + " validation")
        model.eval()
        true_labels = []
        pred_labels = []
        for step, batch in vbar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad(): 
                outputs = model(input_ids=b_input_ids, 
                                                attention_mask=b_input_mask,
                                                labels=b_labels)
                total_eval_loss += outputs.loss.item()
                logits = outputs.logits.detach().cpu().numpy().tolist()
                label_ids = b_labels.to('cpu').numpy().tolist()

                true_labels.extend(label_ids)
                pred_labels.extend(np.argmax(logits,axis=1))

        print(classification_report(pred_labels, true_labels))
        f = open("../outputs/"+output_file, 'a')
        f.write(f"\n {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} \n")
        f.write(classification_report(pred_labels, true_labels))
        f.write('\n')
        f.close()
        model.train()

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
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks    
