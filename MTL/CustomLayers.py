from transformers import PretrainedConfig
from transformers import PretrainedModel

class CustomPhobiaConfig(PretrainedConfig):

    def __init__(self,
                 model_type: str = "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
                 encoder_model: str = "roberta",
                 num_labels: int = 3,
                 dropout: float = .5,
                 inner_dim: int = 1024,
                 max_len: int = 512,
#                 unique_label_count: int = 10,
                 **kwargs):
        super(TagPredictionConfig, self).__init__(num_labels=num_labels, **kwargs)
        self.model_type = model_type
        self.encoder_model = encoder_model
        self.dropout = dropout
        self.inner_dim = inner_dim
        self.max_length = max_len
#        self.unique_label_count = unique_label_count
#        self.intent_token = '<intent>'
#        self.snippet_token = '<snippet>'
#        self.columns_used = ['snippet_tokenized', 'canonical_intent', 'tags']

        encoder_config = AutoConfig.from_pretrained(
            self.encoder_model,
        )
        self.vocab_size = encoder_config.vocab_size
        self.eos_token_id = encoder_config.eos_token_id


class CustomPhobiaModel(PreTrainedModel):
    config_class = TagPredictionConfig

    def __init__(self,
                 config: TagPredictionConfig):
        super(TagPredictionModel, self).__init__(config)
        self.config = config
        self.encoder = AutoModel.from_pretrained(self.config.encoder_model)
        self.encoder.resize_token_embeddings(self.config.vocab_size)
        self.dense_1 = nn.Linear(
            self.encoder.config.hidden_size,
            self.config.inner_dim,
            bias=False
        )
        self.dense_2 = nn.Linear(
            self.config.inner_dim,
            self.config.unique_label_count,
            bias=False
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.encoder._init_weights(self.dense_1)
        self.encoder._init_weights(self.dense_2)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            return_dict=None,
            **kwargs):
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # labels=labels,
            return_dict=return_dict,
        )
        hidden_states = encoded[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        encoded_rep = hidden_states[eos_mask, :].view(
            hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        classification_hidden = self.dropout(encoded_rep)
        classification_hidden = torch.tanh(self.dense_1(classification_hidden))
        classification_hidden = self.dropout(classification_hidden)
        logits = torch.sigmoid(self.dense_2(classification_hidden))

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return TagPredictionOutput(
            loss=loss,
            logits=logits
        )