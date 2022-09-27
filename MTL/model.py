import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import transformers
from three_layer_model import CustomPhobiaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from CustomHead import ModifiedClassificationHead

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
            model_name, 
            config=model_config_dict[task_name],
            )
            # if str(task_name)[-6:] == 'phobia':
            #      model.classifier = ModifiedClassificationHead(model_config_dict[task_name])
            #      model.num_labels = 3
            # else:
            #      model.classifier = ModifiedClassificationHead(model_config_dict[task_name])
            #      model.num_labels = 5

            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)

            taskmodels_dict[task_name] = model
            # f = open("models/"+str(task_name)+'_ModelDesign_NormalLayer', 'a')
            # f.write(str(task_name)+'\n')
            # f.write(str(model))
            # f.write('\n')


        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("XLM"):
            return "roberta"
        elif model_class_name.startswith("Custom"):
            return "roberta"
        elif model_class_name.startswith("MPNet"):
            return "mpnet"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        #print(kwargs)
        # if "tam_phobia" == task_name:
        #     task_name = "tam_sentiment"
        # elif "mal_phobia" == task_name:
        #     task_name = "mal_sentiment"
        # elif "eng_tam_phobia" == task_name:
        #     task_name = "tam_sentiment"
        # elif "eng_phobia" == task_name:
        #     task_name = "tam_sentiment"
        #print(task_name)
        f = open("task_order.txt", 'a')
        f.write(task_name+"\n")
        f.close()
        return self.taskmodels_dict[task_name](**kwargs)