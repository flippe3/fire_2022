import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AutoModel, BertForSequenceClassification
import os
from dataset import MTL_Dataset
import transformers
from model import MultitaskModel
from data_trainer import *
from datasets import load_dataset
from three_layer_model import CustomPhobiaModel
import nlp

LEARNING_RATE = 3e-5

EPOCHS = 4
BATCH_SIZE = 24
os.environ["CUDA_VISIBLE_DEVICES"]="6"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

m_name = "/home/filnil/fire_2022/MTL/MTL_ALL_CUSTOM_HEAD_WEIGHTS"
t_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
filename = "TESTING_STUFF"

tokenizer = AutoTokenizer.from_pretrained(t_name)

model = AutoModel.from_pretrained(m_name, num_labels=3, output_attentions=False)