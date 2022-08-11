from transformers import AutoModelForSequenceClassification
sentences = ["This is an example sentence", "Each sentence is converted"]

model = AutoModelForSequenceClassification.from_pretrained('roberta-base')

print(model.get_encoder_attr_name)

getattr(model, "roberta")