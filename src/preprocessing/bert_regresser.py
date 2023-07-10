import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel

class BertRegresser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        # self.cls_layer1 = nn.Linear(config.hidden_size,128)
        # self.relu1 = nn.ReLU()
        # self.ff1 = nn.Linear(128,32)
        # self.tanh1 = nn.Tanh()
        # self.ff2 = nn.Linear(32,1)
        self.latent_vector = None

        for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        outputs = outputs.last_hidden_state[:,0,:]
        # output = self.cls_layer1(logits)
        # output = self.relu1(output)
        # output = self.ff1(output)
        # output = self.tanh1(output)
        self.latent_vector = outputs
        # output = self.ff2(output)
        return outputs
