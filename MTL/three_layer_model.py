import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import transformers

class CustomPhobiaModel(nn.Module):
    def __init__(self, base_model):
          super(CustomPhobiaModel, self).__init__()
          self.model = base_model 
          self.l1 = nn.Linear(768, 768)
          self.dp1 = nn.Dropout(0.1)
          self.l2 = nn.Linear(768, 768)
          self.dp2 = nn.Dropout(0.1)
          self.out = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, labels):
          outputs = self.model(
               input_ids=input_ids, 
               attention_mask=attention_mask
          )
          sequence_output = self.dp1(outputs[0])
          sequence_output = self.l1(sequence_output)
          sequence_output = self.dp2(sequence_output)
          logits = self.out(sequence_output)
          

          print(logits.shape)
          return logits