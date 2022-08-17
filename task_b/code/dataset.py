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
    # English
    eng_train = "../data/eng_3_train.tsv"
    eng_val = "../data/eng_3_dev.tsv"
    
    # Tamil
    tam_train = "../data/new_tam_train.tsv"
    tam_val = "../data/tam_3_dev.tsv"
     
    # Malayalam
    mal_train = "../data/new_mal_train.tsv"
    mal_val = "../data/mal_3_dev.csv"

    # English-Tamil
    eng_tam_train = "../data/new_eng_tam_train.tsv"
    eng_tam_val = "../data/eng-tam_3_dev.tsv"
     

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
        else:
            texts = self.read_dataset(train_file, test)
            inputs, masks =  self.tokenize_input(texts, tokenizer)
            dataset = TensorDataset(inputs, masks)
            
        return dataset

    def get_phobia_dataset(self, tokenizer, balance=False):
        eng_train = self.get_dataset(tokenizer, self.eng_train, balance=balance)
        eng_val = self.get_dataset(tokenizer, self.eng_val, balance=balance)
        
        tam_train = self.get_dataset(tokenizer, self.tam_train, balance=balance)
        tam_val = self.get_dataset(tokenizer, self.tam_val, balance=balance)

        mal_train = self.get_dataset(tokenizer, self.mal_train, balance=balance)
        mal_val = self.get_dataset(tokenizer, self.mal_val, balance=balance)

        eng_tam_train = self.get_dataset(tokenizer, self.eng_tam_train, balance=balance)
        eng_tam_val = self.get_dataset(tokenizer, self.eng_tam_val, balance=balance)

        return eng_train, eng_val, tam_train, tam_val, mal_train, mal_val, eng_tam_train, eng_tam_val

    def validation(self, model, tokenizer, device, output_file, dataset, BS=16):
        _, eng_val, _, tam_val, _, mal_val, _, eng_tam_val = self.get_phobia_dataset(tokenizer, balance=False)

        if dataset == 'tam':
            loader = DataLoader(tam_val, sampler = SequentialSampler(tam_val), batch_size=BS)
        elif dataset == 'eng':
            loader = DataLoader(eng_val, sampler = SequentialSampler(eng_val), batch_size=BS) 
        elif dataset == 'mal':
            loader = DataLoader(mal_val, sampler = SequentialSampler(mal_val), batch_size=BS) 
        elif dataset == 'eng_tam':
            loader = DataLoader(eng_tam_val, sampler = SequentialSampler(eng_tam_val), batch_size=BS) 

        print(f"{dataset} validation: {len(loader) * BS}")
        
        vbar = tqdm(enumerate(loader), total=len(loader), desc= dataset + " validation")
        model.eval()
        true_labels = []
        pred_labels = []
        #total_eval_loss = 0
        for step, batch in vbar:
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad(): 
                outputs = model(input_ids=b_input_ids, attention_mask=b_masks,
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
        if path[-4:] == ".tsv":
            df = pd.read_csv(path, '\t')
        else:
            df = pd.read_csv(path)
        
        print(f"path:{path} len: {len(df)}")

        df = df.dropna(subset=['text','category'])
        df = df.drop_duplicates(subset='text')

        print(f"path:{path} NEW len: {len(df)}")
        
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

