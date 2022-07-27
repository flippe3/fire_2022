import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import TOKENIZER_MAPPING, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from dataset import Dataset

TOKENIZER_NAME = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
MODEL_NAME = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
LEARNING_RATE = 3e-5

OUTPUT_FILE = "paraphrase-roberta-mal.md"

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
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, output_attentions=False)
model.to(device)
optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, no_deprecation_warning=True)

data = Dataset()
_, _, _, _, mal_train_2022, _, _, _ = data.get_phobia_dataset(tokenizer, balance=False)

train_dataloader = DataLoader(
            mal_train_2022,
            sampler = RandomSampler(mal_train_2022),
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
            #outputs = model(input_ids=b_input_ids, labels=b_labels) 
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

        data.validation(model, tokenizer, device, output_file=OUTPUT_FILE, BS=BATCH_SIZE, dataset='mal')
    torch.save(model, f"../pickles/task_b_mal.pt")
train()
