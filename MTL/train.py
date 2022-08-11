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

LEARNING_RATE = 3e-5

EPOCHS = 4
BATCH_SIZE = 24
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

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

data = MTL_Dataset()

dataset_dict = {
    'kan_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_kan_train.tsv", 'test': "../task_a/data/kan_sentiment_dev.tsv"}),
    'mal_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_mal_train.tsv", 'test': "../task_a/data/Mal_sentiment_dev.tsv"}),
    'tam_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_tam_train.tsv", 'test': "../task_a/data/tam_sentiment_dev.tsv"}),

    'eng_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/eng_3_train.tsv", 'test': "../task_b/data/eng_3_dev.tsv"}),
    'tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_tam_train.tsv", 'test': "../task_b/data/tam_3_dev.tsv"}),
    'mal_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_mal_train.tsv", 'test': "../task_b/data/mal_3_dev.tsv"}),
    'eng_tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_eng_tam_train.tsv", 'test': "../task_b/data/eng-tam_3_dev.tsv"}),
}

def convert_to_features(example_batch):
    print(example_batch['text'][0])
    features = {}
    for text in example_batch['text']:
        features['inputs'].append(tokenizer.encode_plus(
                                    text,            
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    truncation=True,
                                    return_tensors = 'pt'))
        
    features["labels"] = example_batch["category"]
    print(type(features))
    return features

convert_func_dict = {
    "kan_sentiment": convert_to_features,
    "mal_sentiment": convert_to_features,
    "tam_sentiment": convert_to_features,
    "phobia": convert_to_features,
}

columns_dict = {
    "sentiment": ['input_ids', 'attention_mask', 'labels'],
    "phobia": ['input_ids', 'attention_mask', 'labels'],
}

if model_name.startswith("sen"):
    print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.taskmodels_dict["sentiment"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print(multitask_model.taskmodels_dict["phobia"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print("Architecture maybe works?")
else:
    print("Exercise for the reader: add a check for other model architectures =)")


#print(dataset_dict.items())

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        print(phase_dataset)
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        features_dict[task_name][phase].set_format(
            type="torch", 
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))