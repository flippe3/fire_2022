import torch.nn as nn
import torch

class ModifiedClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x 