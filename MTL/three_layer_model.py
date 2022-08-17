import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer 

class CustomPhobiaModel(nn.Module):
    def __init__(self):
          super(CustomPhobiaModel, self).__init__()
          self.model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
          # add your additional layers here, for example a dropout layer followed by a linear classification head
          print("INITIATING THE MODEL")
          #self.l1 = nn.Linear(768, 768)
          #self.dp1 = nn.Dropout(0.1)
          #self.l2 = nn.Linear(768, 768)
          self.dp2 = nn.Dropout(0.1)
          self.out = nn.Linear(768, 3)

    def forward(self, ids, mask, token_type_ids):
          sequence_output, pooled_output = self.model(
               ids, 
               attention_mask=mask,
               token_type_ids=token_type_ids
          )
          
          # we apply dropout to the sequence output, tensor has shape (batch_size, sequence_length, 768)
#          sequence_output = self.dp1(sequence_output)
    
#          sequence_output = self.l1(sequence_output)
          
          sequence_output = self.dp2(sequence_output)
    
          # next, we apply the linear layer. The linear layer (which applies a linear transformation)
          # takes as input the hidden states of all tokens (so seq_len times a vector of size 768, each corresponding to
          # a single token in the input sequence) and outputs 2 numbers (scores, or logits) for every token
          # so the logits are of shape (batch_size, sequence_length, 2)
          logits = self.out(sequence_output)

          return logits