import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import TOKENIZER_MAPPING, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from dataset import *
from util import create_output

TOKENIZER_NAME = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
MODEL_NAME = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
LEARNING_RATE = 3e-5

OUTPUT_FILE = "paraphrase-roberta.md"

EPOCHS = 4
BATCH_SIZE = 24
os.environ["CUDA_VISIBLE_DEVICES"]="4"

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
optimizer = AdamW(model.parameters(), lr = LEARNING_RATE)

data = Dataset()
tam_train_2022, tam_val_2022, _, _, _, _ = data.get_fire_2022_dataset(tokenizer)
tam_train_2021, _, _, _ = data.get_fire_2021_dataset(tokenizer)

create_output(MODEL_NAME, TOKENIZER_NAME, [data.fire_2022_tam_train, data.fire_2021_tam_train], data.fire_2022_tam_val, LEARNING_RATE, EPOCHS, BATCH_SIZE, OUTPUT_FILE)

train_dataloader = DataLoader(
            tam_train_2022 + tam_train_2021,
            sampler = RandomSampler(tam_train_2022 + tam_train_2021),
            batch_size = BATCH_SIZE)

validation_dataloader = DataLoader(
            tam_val_2022,
            sampler = SequentialSampler(tam_val_2022),
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

        data.fire_validation(model, tokenizer, device, output_file=OUTPUT_FILE, year=2022, BS=16, dataset='tam')

train()
