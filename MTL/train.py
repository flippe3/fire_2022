import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
from transformers import TOKENIZER_MAPPING, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from dataset import MTL_Dataset
import transformers
from model import MultitaskModel
import nlp

# LEARNING_RATE = 3e-5

# EPOCHS = 4
# BATCH_SIZE = 24
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# if torch.cuda.is_available():    
#     device = torch.device("cuda")
#     print('There are %d GPU(s) available.' % torch.cuda.device_count())    
#     print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

# else:
#     print('NO GPU AVAILABLE ERROR')
#     device = torch.device("cpu")

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

# model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
# multitask_model = MultitaskModel.create(
#     model_name=model_name,
#     model_type_dict={
#         "sentiment": transformers.AutoModelForSequenceClassification,
#         "phobia": transformers.AutoModelForSequenceClassification
#     },
#     model_config_dict={
#         "sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
#         "phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3)
#     },
# )

# data = MTL_Dataset()

dataset_dict = {
    'kan_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_kan_train.tsv", 'test': "../task_a/data/kan_sentiment_dev.tsv"}),
    'mal_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_mal_train.tsv", 'test': "../task_a/data/Mal_sentiment_dev.tsv"}),
    'tam_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_tam_train.tsv", 'test': "../task_a/data/tam_sentiment_dev.tsv"}),

    'tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_tam_train.tsv", 'test': "../task_b/data/tam_3_dev.tsv"}),
    'eng_tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_eng_tam_train.tsv", 'test': "../task_b/data/eng-tam_3_dev.tsv"})
}



print(dataset_dict)