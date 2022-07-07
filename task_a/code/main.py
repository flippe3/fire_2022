import torch
from tqdm import tqdm
from util import read_dataset, tokenize_input
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os

training_file = "tam_sentiment"
EPOCHS = 4
BATCH_SIZE = 16
os.environ["CUDA_VISIBLE_DEVICES"]="4"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")


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

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def train():
    for epoch_i in range(0, EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')
        total_train_loss = 0
        model.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='train')


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
            avg_train_loss = total_train_loss / len(train_dataloader)
            mem = torch.cuda.memory_reserved(device)/1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(train_loss=f'{avg_train_loss:0.4f}',
                            gpu_mem=f'{mem:0.2f} GB')           

        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Running Validation...")

        vbar = tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc='valid')


        model.eval()
        total_eval_loss = 0
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
        f = open("output_file1.txt", "a"
        )    
        print(classification_report(pred_labels, true_labels))
        f.write("\n")
        f.write(classification_report(pred_labels, true_labels))
        f.write("\n")
        f.close()
train()
