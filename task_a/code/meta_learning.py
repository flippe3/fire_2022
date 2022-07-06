import torch
from tqdm import tqdm
from util import read_dataset, tokenize_input
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
import learn2learn as l2l

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

training_file = "tam_sentiment"
EPOCHS = 4
BATCH_SIZE = 16

print(f"Training file:{training_file}")
tam_labels_train, tam_texts_train = read_dataset(training_file+"_train.tsv")
tam_labels_dev, tam_texts_dev = read_dataset(training_file+"_dev.tsv")


tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5, output_attentions=True)
model.to(device)
optimizer = AdamW(model.parameters(), lr = 2e-5)

train_input_ids, train_attention_masks =  tokenize_input(tam_texts_train, tokenizer)
val_input_ids, val_attention_masks =  tokenize_input(tam_texts_dev, tokenizer)

train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(tam_labels_train, dtype=torch.long)

val_input_ids = torch.cat(val_input_ids, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
val_labels = torch.tensor(tam_labels_dev, dtype=torch.long)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = BATCH_SIZE)

validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = BATCH_SIZE)

train_meta = l2l.data.MetaDataset(train_dataset)
valid_meta = l2l.data.MetaDataset(val_dataset)

# This is the L2L code im working on

def compute_loss(model, loader):
    pbar = tqdm(enumerate(loader), total=len(loader), desc='idk')
    for step, batch in pbar:
        model.zero_grad()
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)


        outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
        total_loss += outputs.loss.item()

        outputs.loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
        
        avg_loss = total_loss / len(train_dataloader)
        
        mem = torch.cuda.memory_reserved(device)/1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(train_loss=f'{avg_loss:0.4f}',
                        gpu_mem=f'{mem:0.2f} GB')           

    return avg_loss 




malm = l2l.algorithms.MAML(model, lr=0.1)
opt = torch.optim.SGD(maml.parameters(), lr=0.001)

for iteration in range(10):
    opt.zero_grad()
    task_model = maml.clone()  # torch.clone() for nn.Modules
    adaptation_loss = compute_loss(task_model)
    task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
    evaluation_loss = compute_loss(task_model)
    evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
    opt.step()
