from lib2to3.pgen2 import token
from sklearn.metrics import classification_report
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from collections import Counter
#from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tokenizers import normalizers
from tokenizers.normalizers import NFD


class Dataset:
    # Fire 2022
    fire_2022_tam_train = "../data/new_tam_train.tsv"
    fire_2022_tam_val = "../data/tam_sentiment_dev.tsv"
     
    fire_2022_kan_train = "../data/new_kan_train.tsv"
    fire_2022_kan_val = "../data/kan_sentiment_dev.tsv"
     
    fire_2022_mal_train = "../data/new_mal_train.tsv"
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

    def get_dataset(self, tokenizer, train_file, test=False, balance=False):
        if test == False:
            labels, texts = self.read_dataset(train_file, test)
            inputs, masks =  self.tokenize_input(texts, tokenizer)

            #print(f"Before balancing: {Counter(labels)}")
            if balance == True:  
                sme = RandomOverSampler(random_state=3407)
                inputs, labels = sme.fit_resample(inputs, labels) 
                inputs = torch.tensor(inputs, dtype=torch.long)
                print(f"After balancing: {Counter(labels)}")
        
            labels = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(inputs, masks, labels)
            #dataset = TensorDataset(inputs, labels)
        else:
            texts = self.read_dataset(train_file, test)
            inputs, masks =  self.tokenize_input(texts, tokenizer)
            dataset = TensorDataset(inputs, masks)
            
        return dataset

    def get_fire_2022_dataset(self, tokenizer, balance=False):
        tam_train = self.get_dataset(tokenizer, self.fire_2022_tam_train, balance=balance)
        tam_val = self.get_dataset(tokenizer, self.fire_2022_tam_val, balance=balance)

        kan_train = self.get_dataset(tokenizer, self.fire_2022_kan_train, balance=balance)
        kan_val = self.get_dataset(tokenizer, self.fire_2022_kan_val, balance=balance)

        mal_train = self.get_dataset(tokenizer, self.fire_2022_mal_train, balance=balance)
        mal_val = self.get_dataset(tokenizer, self.fire_2022_mal_val, balance=balance)

        return tam_train, tam_val, kan_train, kan_val, mal_train, mal_val
         
    def get_fire_2021_dataset(self, tokenizer):
        tam_train = self.get_dataset(tokenizer, self.fire_2021_tam_train)
        tam_test = self.get_dataset(tokenizer, self.fire_2021_tam_val, test=True)

        mal_train = self.get_dataset(tokenizer, self.fire_2021_mal_train)
        mal_test = self.get_dataset(tokenizer, self.fire_2021_mal_val, test=True)

        return tam_train, tam_test, mal_train, mal_test
    
    def get_fire_2020_dataset(self, tokenizer):
        tam_train = self.get_dataset(tokenizer, self.fire_2020_tam_train)
        tam_val = self.get_dataset(tokenizer, self.fire_2020_tam_val)

        mal_train = self.get_dataset(tokenizer, self.fire_2020_mal_train)
        mal_val = self.get_dataset(tokenizer, self.fire_2020_mal_val)

        return tam_train, tam_val, mal_train, mal_val


    def fire_validation(self, model, tokenizer, device, output_file, dataset, year=2022, BS=16):
        if year == 2022:
            _, tam_val, _, kan_val, _, mal_val = self.get_fire_2022_dataset(tokenizer, balance=False)
        elif year == 2021:
           _, tam_val, _, mal_val = self.get_fire_2021_dataset(tokenizer, balance=False)
        elif year == 2020:
           _, tam_val, _, mal_val = self.get_fire_2020_dataset(tokenizer, balance=False)


        if dataset == 'tam':
            loader = DataLoader(tam_val, sampler = SequentialSampler(tam_val), batch_size=BS)
        elif dataset == 'kan' and year == 2022:
            loader = DataLoader(kan_val, sampler = SequentialSampler(kan_val), batch_size=BS) 
        elif dataset == 'mal':
            loader = DataLoader(mal_val, sampler = SequentialSampler(mal_val), batch_size=BS) 

        print(f"{dataset} validation: {len(loader) * BS}")
        
        vbar = tqdm(enumerate(loader), total=len(loader), desc= dataset + " validation")
        model.eval()
        true_labels = []
        pred_labels = []
        #total_eval_loss = 0
        for step, batch in vbar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad(): 
                outputs = model(input_ids=b_input_ids,attention_mask=b_input_mask, 
                                                labels=b_labels)
                #total_eval_loss += outputs.loss.item()
                logits = outputs.logits.detach().cpu().numpy().tolist()
                label_ids = b_labels.to('cpu').numpy().tolist()

                true_labels.extend(label_ids)
                pred_labels.extend(np.argmax(logits,axis=1))

        print(classification_report(pred_labels, true_labels))
        f = open("../outputs/"+output_file, 'a')
        f.write(f"\n {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} \n")
        f.write("```\n")
        f.write(classification_report(pred_labels, true_labels))
        f.write("```\n")
        f.close()
        model.train()


    def read_dataset(self, path, test=False):
        df = pd.read_csv(path, '\t')
        texts = df.text.values

        if test == False:
            label_cats = df.category.astype('category').cat
            label_names = label_cats.categories
            labels = label_cats.codes

            #print("Texts:", len(texts))
            #print("Label names:", label_names)
            return labels, texts
        else:
            return texts

    def tokenize_input(self, texts, tokenizer):        
        input_ids = []
        attention_masks = []
        normalizer = normalizers.Sequence([NFD()])

        for text in texts:
            text = normalizer.normalize_str(text)
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
