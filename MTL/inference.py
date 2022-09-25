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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import nlp

LEARNING_RATE = 3e-5

EPOCHS = 4
BATCH_SIZE = 24
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

m_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

filename = "TESTING_STUFF"

tokenizer = AutoTokenizer.from_pretrained(m_name)

model_name = m_name

multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "kan_sentiment": transformers.AutoModelForSequenceClassification,
        "mal_sentiment": transformers.AutoModelForSequenceClassification,
        "tam_sentiment": transformers.AutoModelForSequenceClassification,
        "eng_phobia": transformers.AutoModelForSequenceClassification,
        "tam_phobia": transformers.AutoModelForSequenceClassification, 
        "mal_phobia": transformers.AutoModelForSequenceClassification, 
        "eng_tam_phobia": transformers.AutoModelForSequenceClassification, 
    },
    model_config_dict={
        "kan_sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        "mal_sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        "tam_sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        "eng_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
        "mal_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
        "tam_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
        "eng_tam_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
    },
)

# dataset_dict = {
#     'kan_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_kan_train.tsv", 'test': "../task_a/data/NEW_KAN_TEST.tsv"}),
#     'mal_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_mal_train.tsv", 'test': "../task_a/data/NEW_MAL_TEST.tsv"}),
#     'tam_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_tam_train.tsv", 'test': "../task_a/data/NEW_TAM_TEST.tsv"}),

#     'eng_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/eng_3_train.tsv", 'test': "../task_b/data/NEW_ENG_TEST.tsv"}),
#     'tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_tam_train.tsv", 'test': "../task_b/data/NEW_TAM_TEST.tsv"}),
#     'mal_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_mal_train.tsv", 'test': "../task_b/data/NEW_MAL_TEST.tsv"}),
#     'eng_tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_eng_tam_train.tsv", 'test': "../task_b/data/NEW_TAM_ENG_TEST.tsv"}),
# }

dataset_dict = {
    'kan_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_kan_train.tsv", 'test': "../task_a/data/kan_sentiment_dev.tsv"}),
    'mal_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_mal_train.tsv", 'test': "../task_a/data/Mal_sentiment_dev.tsv"}),
    'tam_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_tam_train.tsv", 'test': "../task_a/data/tam_sentiment_dev.tsv"}),

    'eng_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/eng_3_train.tsv", 'test': "../task_b/data/eng_3_dev.tsv"}),
    'tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_tam_train.tsv", 'test': "../task_b/data/tam_3_dev.tsv"}),
    'mal_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_mal_train.tsv", 'test': "../task_b/data/mal_3_dev.tsv"}),
    'eng_tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_eng_tam_train.tsv", 'test': "../task_b/data/eng-tam_3_dev.tsv"}),
}


def convert_to_mal(example_batch):
    features = {}
    features = tokenizer.batch_encode_plus(
                                    example_batch['text'],            
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    truncation=True)
    if 'category' in list(example_batch.keys()):
        new_labels = []
        for i in example_batch['category']:
            if i == "Positive":
                new_labels.append(0)
            elif i == "Negative":
                new_labels.append(1)
            elif i == "not-malayalam":
                new_labels.append(2)
            elif i == "unknown_state":
                new_labels.append(3)
            elif i == "Mixed_feelings":
                new_labels.append(4)
            else:
                print("Error", i, len(i))
        features["labels"] = new_labels
    return features
    
def convert_to_kan(example_batch):
    features = {}
    features = tokenizer.batch_encode_plus(
                                    example_batch['text'],            
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    truncation=True)
    if 'category' in list(example_batch.keys()):
        new_labels = []
        for i in example_batch['category']:
            if i == "Positive":
                new_labels.append(0)
            elif i == "Negative":
                new_labels.append(1)
            elif i == "not-Kannada":
                new_labels.append(2)
            elif i == "unknown state":
                new_labels.append(3)
            elif i == "Mixed feelings":
                new_labels.append(4)
            else:
                print("Error", i)

        features["labels"] = new_labels 
    return features

def convert_to_tam(example_batch):
    features = {}
    features = tokenizer.batch_encode_plus(
                                    example_batch['text'],            
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    truncation=True)
    if 'category' in list(example_batch.keys()):
        new_labels = []
        for i in example_batch['category']:
            if i == "Positive":
                new_labels.append(0)
            elif i == "Negative":
                new_labels.append(1)
            elif i == "not-Tamil":
                new_labels.append(2)
            elif i == "unknown_state":
                new_labels.append(3)
            elif i == "Mixed_feelings":
                new_labels.append(4)
            else:
                print("Error", i)

        features["labels"] = new_labels 
    return features

def convert_to_phobia(example_batch):
    features = {}
    features = tokenizer.batch_encode_plus(
                                    example_batch['text'],            
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    truncation=True)
    if 'category' in list(example_batch.keys()):
        new_labels = []
        for i in example_batch['category']:
            if i == "Non-anti-LGBT+ content":
                new_labels.append(0)
            elif i == "Homophobic":
                new_labels.append(1)
            elif i == "Transphobic":
                new_labels.append(2)
            else:
                print("Error", i)

        features["labels"] = new_labels 
    return features
    
convert_func_dict = {
    "kan_sentiment": convert_to_kan,
    "mal_sentiment": convert_to_mal,
    "tam_sentiment": convert_to_tam,
    "eng_phobia": convert_to_phobia,
    "tam_phobia": convert_to_phobia,
    "mal_phobia": convert_to_phobia,
    "eng_tam_phobia": convert_to_phobia,
}

columns_dict = {
    "kan_sentiment": ['input_ids', 'attention_mask', 'labels'],
    "mal_sentiment": ['input_ids', 'attention_mask', 'labels'],
    "tam_sentiment": ['input_ids', 'attention_mask', 'labels'],
    
    "eng_phobia": ['input_ids', 'attention_mask', 'labels'],
    "tam_phobia": ['input_ids', 'attention_mask', 'labels'],
    "mal_phobia": ['input_ids', 'attention_mask', 'labels'],
    "eng_tam_phobia": ['input_ids', 'attention_mask', 'labels'],
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False
        )
        features_dict[task_name][phase].set_format(
            type="torch", 
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

train_dataset = {
		task_name: dataset["train"] for task_name, dataset in features_dict.items()
}

val_dataset = {
		task_name: dataset["test"] for task_name, dataset in features_dict.items()
}

args = transformers.TrainingArguments(
    save_strategy="no",
    output_dir="output_trainer",
    overwrite_output_dir=True,
    learning_rate=LEARNING_RATE,
    num_train_epochs=4,
#        warmup_steps=500,
    per_device_train_batch_size=24,
)


print("*******************")
print("STARTING TRAINING")
trainer = MultitaskTrainer(
    model=multitask_model,
    data_collator=NLPDataCollator(),
    train_dataset=train_dataset,
    args=args 
)
trainer.train()
#trainer.save_model()
#trainer.save_state()

#loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

preds_dict = {}
for task_name in ["tam_sentiment", "kan_sentiment","mal_sentiment", "eng_phobia", "tam_phobia", "mal_phobia", "eng_tam_phobia"]:
#for task_name in ["kan_sentiment", "tam_sentiment", "mal_sentiment"]:
    print("Starting validation", task_name)

    val_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["test"])
    )
    # test_dataloader = DataLoaderWithTaskname(
    #     task_name,
    #     trainer.get_eval_dataloader(eval_dataset=test_features_dict[task_name]['test'])
    # )

    preds_dict[task_name] = trainer.evaluation_loop(val_dataloader, description='test')
    #preds_dict[task_name] = trainer.evaluate(test_dataloader)
print(preds_dict) 
from sklearn.metrics import classification_report
# 0 = Positive
# 1 = Negative
# 2 = not-Kannada
# 3 = unknown state
# 4 = Mixed feelings
# s_kan_ids = pd.read_csv('../task_a/data/kan_test.tsv', '\t')['id']
# s_tam_ids = pd.read_csv('../task_a/data/tam_test.tsv', '\t')['id']
# s_mal_ids = pd.read_csv('../task_a/data/mal_test.tsv', '\t')['id']

# t_eng_ids = pd.read_csv('../task_b/data/eng_test.tsv', '\t')['id']
# t_tam_ids = pd.read_csv('../task_b/data/tam_test.tsv', '\t')['id']
# t_mal_ids = pd.read_csv('../task_b/data/mal_test.tsv', '\t')['id']
# t_eng_tam_ids = pd.read_csv('../task_b/data/tam-eng_test.tsv', '\t')['id']

f = open('test_output3/kan_sentiment', 'w')
preds = np.argmax(preds_dict['kan_sentiment'].predictions ,axis=1)
count = 0
for i in preds:
    if i == 0:
        f.write(f'Positive\n')
    elif i == 1:
        f.write(f'Negative\n')
    elif i == 2:
        f.write(f'not-Kannada\n')
    elif i == 3:
        f.write(f'unknown state\n')
    elif i == 4:
        f.write(f'Mixed feelings\n')
    count += 1
# 0 = Positive
# 1 = Negative
# 2 = not-Tamil
# 3 = unknown_state
# 4 = Mixed_feelings
f = open('test_output3/tam_sentiment', 'w')
preds = np.argmax(preds_dict['tam_sentiment'].predictions ,axis=1)
count = 0
for i in preds:
    if i == 0:
        f.write(f'Positive\n')
    elif i == 1:
        f.write(f'Negative\n')
    elif i == 2:
        f.write(f'not-Tamil\n')
    elif i == 3:
        f.write(f'unknown_state\n')
    elif i == 4:
        f.write(f'Mixed_feelings\n')
    count += 1

# 0 = Positive
# 1 = Negative
# 2 = not-malayalam
# 3 = unknown_state
# 4 = Mixed_feelings
f = open('test_output3/mal_sentiment', 'w')
preds = np.argmax(preds_dict['mal_sentiment'].predictions ,axis=1)
count = 0
for i in preds:
    if i == 0:
        f.write(f'Positive\n')
    elif i == 1:
        f.write(f'Negative\n')
    elif i == 2:
        f.write(f'not-malayalam\n')
    elif i == 3:
        f.write(f'unknown_state\n')
    elif i == 4:
        f.write(f'Mixed_feelings\n')
    count += 1

# 0 = Non-anti-LGBT+ content 
# 1 = Homophobic
# 2 = Transphobic
f = open('test_output3/tam_phobia', 'w')
preds = np.argmax(preds_dict['tam_phobia'].predictions ,axis=1)
count = 0 
for i in preds:
    if i == 0:
        f.write(f'Non-anti-LGBT+ content\n')
    elif i == 1:
        f.write(f'Homophobic\n')
    elif i == 2:
        f.write(f'Transphobic\n')
    count += 1

f = open('test_output3/eng_phobia', 'w')
preds = np.argmax(preds_dict['eng_phobia'].predictions ,axis=1)
count = 0
for i in preds:
    if i == 0:
        f.write(f'Non-anti-LGBT+ content\n')
    elif i == 1:
        f.write(f'Homophobic\n')
    elif i == 2:
        f.write(f'Transphobic\n')
    count += 1

f = open('test_output3/mal_phobia', 'w')
preds = np.argmax(preds_dict['mal_phobia'].predictions ,axis=1)
count = 0
for i in preds:
    if i == 0:
        f.write(f'Non-anti-LGBT+ content\n')
    elif i == 1:
        f.write(f'Homophobic\n')
    elif i == 2:
        f.write(f'Transphobic\n')
    count += 1

f = open('test_output3/eng_tam_phobia', 'w')
preds = np.argmax(preds_dict['eng_tam_phobia'].predictions ,axis=1)
count = 0
for i in preds:
    if i == 0:
        f.write(f'Non-anti-LGBT+ content\n')
    elif i == 1:
        f.write(f'Homophobic\n')
    elif i == 2:
        f.write(f'Transphobic\n')
    count += 1