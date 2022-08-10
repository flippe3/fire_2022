import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import TOKENIZER_MAPPING, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from dataset import MTL_Dataset
from model import MLTModel

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

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
model = MLTModel()
model.to(device)

model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "sentiment": transformers.AutoModelForSequenceClassification,
        "phobia": transformers.AutoModelForSequenceClassification
    },
    model_config_dict={
        "sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        "phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3)
    },
)