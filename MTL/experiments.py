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
import random

# Reproducibility
random_seed = 3407
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

LEARNING_RATE = 3e-5

EPOCHS = 4
BATCH_SIZE = 24

os.environ["CUDA_VISIBLE_DEVICES"]="3,4"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

m_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

filename = "MTL_ALL_CUSTOM_NORMAL"

tokenizer = AutoTokenizer.from_pretrained(m_name)

#custom_phobia_model = CustomPhobiaModel()
#print(custom_phobia_model)
model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "sentiment": transformers.AutoModelForSequenceClassification,
        "phobia": transformers.AutoModelForSequenceClassification,
        # "mal_sentiment": transformers.AutoModelForSequenceClassification,
        # "kan_sentiment": transformers.AutoModelForSequenceClassification,
        # "tam_sentiment": transformers.AutoModelForSequenceClassification,
        # "eng_phobia": transformers.AutoModelForSequenceClassification,
        # "tam_phobia": transformers.AutoModelForSequenceClassification, 
        # "mal_phobia": transformers.AutoModelForSequenceClassification, 
        # "eng_tam_phobia": transformers.AutoModelForSequenceClassification, 
    },
    model_config_dict={
        "sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        "phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3),
        # "kan_sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        # "mal_sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        # "tam_sentiment": transformers.AutoConfig.from_pretrained(model_name, num_labels=5),
        # "eng_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
        # "mal_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
        # "tam_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
        # "eng_tam_phobia": transformers.AutoConfig.from_pretrained(model_name, num_labels=3), 
    },
)

#print(custom_phobia_model)
#print(transformers.AutoConfig.from_pretrained(model_name, num_labels=3))

dataset_dict = {
    # 'sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': ["../task_a/data/FAKE.tsv"]}),
    # 'phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': ["../task_b/data/FAKE.tsv"]}),
    # 'sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': ["../task_a/data/new_kan_train.tsv", "../task_a/data/new_tam_train.tsv", "../task_a/data/new_mal_train.tsv"]}),
    # 'phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': ["../task_b/data/eng_3_train.tsv", "../task_b/data/new_tam_train.tsv", "../task_b/data/new_mal_train.tsv", "../task_b/data/new_eng_tam_train.tsv"]}),
    'sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': ["../task_a/data/new_tam_train.tsv"]}),
    'phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': ["../task_b/data/eng_3_train.tsv", "../task_b/data/new_tam_train.tsv", "../task_b/data/new_eng_tam_train.tsv"]}),
    'kan_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_kan_train.tsv", 'test': "../task_a/data/test/new_kan_test.tsv", 'validation': "../task_a/data/kan_sentiment_dev.tsv"}),
    'mal_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_mal_train.tsv", 'test': "../task_a/data/test/new_mal_test.tsv", 'validation': "../task_a/data/Mal_sentiment_dev.tsv"}),
    'tam_sentiment': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_tam_train.tsv", 'test': "../task_a/data/test/new_tam_test.tsv", 'validation': "../task_a/data/tam_sentiment_dev.tsv"}),

    'eng_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/eng_3_train.tsv", 'test': "../task_b/data/test/new_eng_test.tsv", 'validation': "../task_b/data/eng_3_dev.tsv"}),
    'tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_tam_train.tsv", 'test': "../task_b/data/test/new_tam_test.tsv", 'validation': "../task_b/data/tam_3_dev.tsv"}),
    'mal_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_mal_train.tsv", 'test': "../task_b/data/test/new_mal_test.tsv", 'validation': "../task_b/data/mal_3_dev.tsv"}),
    'eng_tam_phobia': nlp.load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_eng_tam_train.tsv", 'test': "../task_b/data/test/new_tam_eng_test.tsv", 'validation': "../task_b/data/eng-tam_3_dev.tsv"}),
}

def convert_to_sentiment(example_batch):
    features = {}
    features = tokenizer.batch_encode_plus(
                                    example_batch['text'],            
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    truncation=True)
    new_labels = []
    for i in example_batch['category']:
        if i == "Positive" or i == "positive":
            new_labels.append(0)
        elif i == "Negative" or i == "negative":
            new_labels.append(1)
        elif i == "not-malayalam" or i == "not-Kannada" or i == "not-Tamil":
            new_labels.append(2)
        elif i == "unknown_state" or i == "neutral" or i == "unknown state":
            new_labels.append(3)
        elif i == "Mixed_feelings" or i == "mixed" or i == "Mixed feelings":
            new_labels.append(4)
        else:
            print("Error", i, len(i))
    features["labels"] = new_labels
    return features
    
def convert_to_phobia(example_batch):
    features = {}
    print(len(example_batch['text']))
    features = tokenizer.batch_encode_plus(
                                    example_batch['text'],            
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    truncation=True)
    new_labels = []
    for i in example_batch['category']:
        if i == "Non-anti-LGBT+ content" or i == "Non- anti LGBT content":
            new_labels.append(0)
        elif i == "Homophobic" or i == "homophobia":
            new_labels.append(1)
        elif i == "Transphobic" or i == "transphobia":
            new_labels.append(2)
        else:
            print("Error", i)

    features["labels"] = new_labels 
    return features
    
convert_func_dict = {
    "sentiment": convert_to_sentiment,
    "phobia": convert_to_phobia,
    "kan_sentiment": convert_to_sentiment,
    "mal_sentiment": convert_to_sentiment,
    "tam_sentiment": convert_to_sentiment,
    "eng_phobia": convert_to_phobia,
    "tam_phobia": convert_to_phobia,
    "mal_phobia": convert_to_phobia,
    "eng_tam_phobia": convert_to_phobia,
}

columns_dict = {
    "sentiment": ['input_ids', 'attention_mask', 'labels'],
    "phobia": ['input_ids', 'attention_mask', 'labels'],
    "mal_sentiment": ['input_ids', 'attention_mask', 'labels'],
    "kan_sentiment": ['input_ids', 'attention_mask', 'labels'],
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
            load_from_cache_file=False,
        )
        features_dict[task_name][phase].set_format(
            type="torch", 
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

# train_dataset = {
# 		task_name: dataset["train"] for task_name, dataset in features_dict.items()
# }

train_dataset = {
    'sentiment': features_dict['sentiment']['train'],
    'phobia': features_dict['phobia']['train']
}

print(train_dataset)

print("STARTING TRAINING")
trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        save_strategy="no",
        output_dir="output_trainer",
        overwrite_output_dir=True,
        learning_rate=LEARNING_RATE,
        do_train=True,
        num_train_epochs=4,
        per_device_train_batch_size=BATCH_SIZE,
    ),
    data_collator=NLPDataCollator(),
    train_dataset=train_dataset,
)
trainer.train()
print("FINISHED TRAINING")
#trainer.evaluate()



preds_dict = {}
val_dict = {}

# Experiment 1
# tasks = ["tam_phobia", "mal_phobia", "eng_phobia", "eng_tam_phobia"]

# Experiment 2
#tasks = ["mal_sentiment", "tam_sentiment", "kan_sentiment", "tam_phobia", "mal_phobia", "eng_phobia", "eng_tam_phobia"]

# Experiment 3.1
tasks = ["tam_sentiment", "eng_phobia", "eng_tam_phobia", "tam_phobia"]

for task_name in tasks:
    print("Starting validation", task_name)
    eval_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["test"])
    )
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer.evaluation_loop(
        eval_dataloader,
        description=f"Test: {task_name}",
    )

    validation_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["validation"])
    )

    print(validation_dataloader.data_loader.collate_fn)
    val_dict[task_name] = trainer.evaluation_loop(
        validation_dataloader,
        description=f"Validation: {task_name}",
    )

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def write_output(filename, preds_dict, val_dict, features_dict):
    f = open('new_outputs/'+filename, 'w')
    
    for task in tasks:
        preds = np.argmax(preds_dict[task].predictions ,axis=1)
        vals = np.argmax(val_dict[task].predictions ,axis=1)
        ground_truth = features_dict[task]['test']['labels']

        f.write(task + " Test\n")
        f.write(classification_report(preds, ground_truth))
        f.write(task + "Val")
        ground_truth = features_dict[task]['validation']['labels']
        f.write(classification_report(vals, ground_truth))    
        # f.write(str(confusion_matrix(ground_truth, preds)))
        f.write(str(confusion_matrix(ground_truth, vals)))
        f.write("-----------------------------------------\n")

    f.close()

    print("Successfully wrote file")

write_output("experiment 5.2.1", preds_dict, val_dict, features_dict)