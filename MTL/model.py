import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class MLTModel(nn.Module):
    def __init__(self):
        super(MLTModel, self).__init__()
        
        self.base_model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        self.dropout = nn.Dropout(0.5)
        self.sentiment_classifier = nn.Linear(768, 5)
        self.phobia_classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, labels, task_name):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels = labels)
        # You write you new head here
        #outputs = self.dropout(outputs[0])
        
        if task_name == "sentiment":
            outputs = self.sentiment_classifier(outputs.logits)
        elif task_name == "phobia":
            outputs = self.phobia_classifier(outputs.logits)
       
        return outputs