import pandas as pd

def read_dataset(path):
    df = pd.read_csv('../data/' + path, '\t')
    texts = df.text.values
    label_cats = df.category.astype('category').cat
    label_names = label_cats.categories
    labels = label_cats.codes

    print("Texts:", len(texts))
    print("Label names:", label_names)
    return labels, texts

def tokenize_input(texts, tokenizer):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,            
                            add_special_tokens = True,
                            max_length = 512,
                            padding = 'max_length',
                            return_attention_mask = True,
                            truncation=True,
                            return_tensors = 'pt')
    
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks