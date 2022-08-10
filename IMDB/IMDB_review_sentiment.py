import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import TOKENIZER_MAPPING, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from datasets import load_dataset

TOKENIZER_NAME = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
MODEL_NAME = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
LEARNING_RATE = 3e-5

EPOCHS = 4
BATCH_SIZE = 24 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')
else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")
   
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5, output_attentions=True)
model.to(device)
optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, no_deprecation_warning=True)

data = pd.read_csv('train-2.tsv', '\t')
data = data.sample(frac=1)
dataset_train, dataset_val = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
print(len(dataset_train), len(dataset_val))
print(dataset_val)

def tokenize_input(texts, tokenizer):        
    input_ids = []
    attention_masks = []
    #normalizer = normalizers.Sequence([NFD()])

    for text in texts:
        #text = normalizer.normalize_str(text)
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

print("Loading dataset")
inputs_train, masks_train = tokenize_input(dataset_train.Phrase.values, tokenizer)
labels_train = torch.tensor(dataset_train.Sentiment.values, dtype=torch.long)

inputs_val, masks_val = tokenize_input(dataset_val.Phrase.values, tokenizer)
labels_val = torch.tensor(dataset_val.Sentiment.values, dtype=torch.long)

train_dataset = TensorDataset(inputs_train, masks_train, labels_train)
val_dataset = TensorDataset(inputs_val, masks_val, labels_val)

dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE)
dataloader_val = DataLoader(val_dataset, batch_size=BATCH_SIZE)
print("Loaded dataset")
total_steps = len(dataloader_train) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def train():
    for epoch_i in range(0, EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')
        total_train_loss = 0
        model.train()

        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc='train')

        for step, batch in pbar:
            model.zero_grad()        

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
            total_train_loss += outputs.loss.item()

            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            avg_train_loss = total_train_loss / len(dataloader_train)
            mem = torch.cuda.memory_reserved(device)/1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(train_loss=f'{avg_train_loss:0.4f}',
                            gpu_mem=f'{mem:0.2f} GB')           

    
        vbar = tqdm(enumerate(dataloader_val), total=len(dataloader_val), desc=" validation")

        model.eval()
        
        true_labels = []
        pred_labels = []
        #total_eval_loss = 0
        
        # Label names: Index(['Mixed_feelings', 'Negative', 'Positive', 'not-Tamil', 'unknown_state']
        # Label names: Index(['Homophobic', 'Non-anti-LGBT+ content', 'Transphobic']
        
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
    model.save_pretrained("./pickles_eng")
train()
