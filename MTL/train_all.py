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

LEARNING_RATE = 3e-5

EPOCHS = 4
BATCH_SIZE = 12 
os.environ["CUDA_VISIBLE_DEVICES"]="5"

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())    
    print(f'We will use the GPU:{torch.cuda.get_device_name()} ({device})')

else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

#custom_phobia_model = CustomPhobiaModel()
#print(custom_phobia_model)
model_name = "xlm-roberta-large"

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
#print(custom_phobia_model)
#print(transformers.AutoConfig.from_pretrained(model_name, num_labels=3))

dataset_dict = {
    'kan_sentiment': load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_kan_train.tsv", 'test': "../task_a/data/kan_sentiment_dev.tsv"}),
    'mal_sentiment': load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_mal_train.tsv", 'test': "../task_a/data/Mal_sentiment_dev.tsv"}),
    'tam_sentiment': load_dataset('csv', delimiter='\t', data_files={'train': "../task_a/data/new_tam_train.tsv", 'test': "../task_a/data/tam_sentiment_dev.tsv"}),

    'eng_phobia': load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/eng_3_train.tsv", 'test': "../task_b/data/eng_3_dev.tsv"}),
    'tam_phobia': load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_tam_train.tsv", 'test': "../task_b/data/tam_3_dev.tsv"}),
    'mal_phobia': load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_mal_train.tsv", 'test': "../task_b/data/mal_3_dev.tsv"}),
    'eng_tam_phobia': load_dataset('csv', delimiter='\t', data_files={'train': "../task_b/data/new_eng_tam_train.tsv", 'test': "../task_b/data/eng-tam_3_dev.tsv"}),
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
            load_from_cache_file=False,
        )
        features_dict[task_name][phase].set_format(
            type="torch", 
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

train_dataset = {
		task_name: dataset["train"] for task_name, dataset in features_dict.items()
}

trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        save_strategy="no",
        output_dir="output_trainer",
        overwrite_output_dir=True,
        learning_rate=1e-5,
        do_train=True,
        num_train_epochs=4,
        per_device_train_batch_size=BATCH_SIZE,
    ),
    data_collator=NLPDataCollator(),
    train_dataset=train_dataset,
)
trainer.train()

trainer.save_model("ALL_XLM_ROBERTA_LARGE")

preds_dict = {}
for task_name in ["tam_sentiment", "kan_sentiment","mal_sentiment", "eng_phobia", "tam_phobia", "mal_phobia", "eng_tam_phobia"]:
    print("Starting validation", task_name)
    eval_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["test"])
    )
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer.evaluation_loop(
        eval_dataloader,
        description=f"Validation: {task_name}",
    )

from sklearn.metrics import classification_report

f = open('output_all_xlm_roberta_large', 'w')

preds = np.argmax(preds_dict['tam_sentiment'].predictions ,axis=1)
ground_truth = features_dict['tam_sentiment']['test']['labels']

f.write("Tam Sentiment\n")
f.write(classification_report(preds, ground_truth))
f.write("-----------------------------------------")
print("Tam Sentiment:\n", classification_report(preds, ground_truth, target_names=['Positive', 'Negative', 'not-lang', 'unknown_state', 'Mixed_feelings']))

preds = np.argmax(preds_dict['kan_sentiment'].predictions ,axis=1)
ground_truth = features_dict['kan_sentiment']['test']['labels']

f.write("Kan Sentiment\n")
f.write(classification_report(preds, ground_truth))
f.write("-----------------------------------------")
print("Kan Sentiment:\n", classification_report(preds, ground_truth, target_names=['Positive', 'Negative', 'not-lang', 'unknown_state', 'Mixed_feelings']))

preds = np.argmax(preds_dict['mal_sentiment'].predictions ,axis=1)
ground_truth = features_dict['mal_sentiment']['test']['labels']

f.write("Mal Sentiment\n")
f.write(classification_report(preds, ground_truth))
f.write("-----------------------------------------")
print("Sentiment:\n", classification_report(preds, ground_truth, target_names=['Positive', 'Negative', 'not-lang', 'unknown_state', 'Mixed_feelings']))

preds = np.argmax(preds_dict['tam_phobia'].predictions ,axis=1)
ground_truth = features_dict['tam_phobia']['test']['labels']

print("Tam Phobia:\n", classification_report(preds, ground_truth, target_names=['Non-Anti-LGBTQ+', 'Homophobic', 'Transphobic']))
f.write("Tam Phobia\n")
f.write(classification_report(preds, ground_truth, target_names=['Non-Anti-LGBTQ+', 'Homophobic', 'Transphobic']))

preds = np.argmax(preds_dict['eng_phobia'].predictions ,axis=1)
ground_truth = features_dict['eng_phobia']['test']['labels']

print("Eng Phobia:\n", classification_report(preds, ground_truth, target_names=['Non-Anti-LGBTQ+', 'Homophobic', 'Transphobic']))
f.write("Eng Phobia\n")
f.write(classification_report(preds, ground_truth, target_names=['Non-Anti-LGBTQ+', 'Homophobic', 'Transphobic']))

preds = np.argmax(preds_dict['mal_phobia'].predictions ,axis=1)
ground_truth = features_dict['mal_phobia']['test']['labels']

print("Mal Phobia:\n", classification_report(preds, ground_truth, target_names=['Non-Anti-LGBTQ+', 'Homophobic', 'Transphobic']))
f.write("Mal Phobia\n")
f.write(classification_report(preds, ground_truth, target_names=['Non-Anti-LGBTQ+', 'Homophobic', 'Transphobic']))

preds = np.argmax(preds_dict['eng_tam_phobia'].predictions ,axis=1)
ground_truth = features_dict['eng_tam_phobia']['test']['labels']

f.write("-----------------------------------------")
f.write("Eng Tam Phobia\n")
f.write(classification_report(preds, ground_truth, target_names=['Non-Anti-LGBTQ+', 'Homophobic', 'Transphobic']))
f.write("-----------------------------------------")
f.close()

print("Eng Tam Phobia:\n", classification_report(preds, ground_truth, target_names=['Positive', 'Negative', 'not-lang', 'unknown_state', 'Mixed_feelings']))
