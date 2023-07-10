# BERT MODEL
import torch as th
import torch.nn as nn
from transformers import BertModel,BertPreTrainedModel, BertTokenizer,\
                         RobertaModel, RobertaPreTrainedModel, RobertaTokenizer 

from bert_regresser import BertRegresser
from transformers import AutoConfig, AutoTokenizer

class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.latent_vector = None
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        self.embedding = pooled_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForMultiLabelClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.embedding_layer = nn.Linear(config.hidden_size, self.embedding_size )
        self.classifier = nn.Linear(self.embedding_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.latent_vector = None
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        self.latent_vector = pooled_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return self.latent_vector  # (loss), logits, (hidden_states), (attentions)

# GoEmo PIPELINE

from typing import Union, Optional

import numpy as np
import pandas as pd
from transformers.pipelines import ArgumentHandler
from transformers import (
    Pipeline,
    PreTrainedTokenizer,
    ModelCard
)


class MultiLabelPipeline(Pipeline):
    def __init__(
            self,
            model: Union["Ber", "PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            args_parser: ArgumentHandler = None,
            device: int = -1,
            binary_output: bool = False,
            threshold: float = 0.3
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task
        )

        self.threshold = threshold

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        scores = 1 / (1 + np.exp(-outputs))  # Sigmoid
        scores = np.round(scores, 4)
        return scores


class LatentPipeline(MultiLabelPipeline):
    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        embedding = self.model.embedding.cpu()
        embedding = np.round(embedding, 4)
        return embedding


class FineTunedLatentPipeline(Pipeline):
    def __init__(
            self,
            model: Union["BertRegresser", "PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: AutoTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            args_parser: ArgumentHandler = None,
            device: int = -1,
            binary_output: bool = False,
            threshold: float = 0.3
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task
        )

        self.threshold = threshold

    def _forward(self, inputs, return_tensors=False):
        input_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        embedding = self.model.latent_vector.cpu()
        embedding = np.round(embedding.detach().numpy(), 4)
        return embedding

# class FineTunedLatentPipeline(MultiLabelPipeline):
#     def __call__(self, *args, **kwargs):
#         outputs = super().__call__(*args, **kwargs)
#         embedding = self.model.latent_vector.cpu()
#         embedding = np.round(embedding, 4)
#         return embedding


from tqdm import tqdm
from collections import defaultdict

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config import *

GPU = -1
if th.cuda.is_available():
    GPU = 0

class FeatureExtractor:
    def __init__(self, gpu=0) -> None:
        self.pretrained = { 
            "GoEmo" : "monologg/bert-base-cased-goemotions-original",
            "GoEmoLatent" : "monologg/bert-base-cased-goemotions-original",
            "RoBERTaSentiment" : "siebert/sentiment-roberta-large-english",
            "RoBERTaSentimentLatent" : "siebert/sentiment-roberta-large-english",
            "BERTBase" : "bert-base-uncased",
            "BERTBaseLatent" : "bert-base-uncased",
            "BertFineTunedLatent" : "bert-large-uncased",
        }

        self.finetuned ={
            "BertFineTunedLatent" : './bert_models/fintune_bert_1478.pt' 
        }

        self.gpu = gpu

    def add_feature_vector(self, df, data_path, text_col, feature, isolate_feature=False):
        # self._set_tokenizer(feature)

        if 'FineTuned' in feature:
            self._set_finetuned_model(feature)
            pipeline = FineTunedLatentPipeline(model=self.model, tokenizer=self.tokenizer, device=self.gpu)
        elif 'Latent' in feature:
            self._set_model(feature)
            pipeline = LatentPipeline(model=self.model, tokenizer=self.tokenizer, device=self.gpu)
        else:
            self._set_model(feature)
            pipeline = MultiLabelPipeline(model=self.model, tokenizer=self.tokenizer, device=self.gpu)
        

        batch_size = 8 ###256
        review_texts = df[text_col]
        batchs = [ review_texts[i:i+batch_size] for i in range(0,len(review_texts),batch_size)]

        feature_array = None
        for batch in tqdm(batchs):
            batch = batch.tolist()
            # batch = [ self._length_limit(text) for text in batch]
            outputs = pipeline(batch)
            # (batch, dim)
            
            if feature_array is None:
                feature_array = outputs
            else:
                feature_array = np.concatenate([feature_array, outputs], axis=0)


        edge_feature_file = None 
        # total feature vectors (n, dim)
        if isolate_feature==False:
            feature_dict={}
            for k in range(feature_array.shape[1]):
                feature_dict[f'{feature}_{k}'] = feature_array[:,k]
            
            feature_df = pd.DataFrame(feature_dict)

            edge_feature_file = f'{data_path}_feature.csv'
            feature_df.to_csv(edge_feature_file)
            # df = pd.concat([df,feature_df], axis=1)
        else:
            #save numpy array as file
            edge_feature_file = f'{data_path}_feature_array.npy'
            with open(edge_feature_file, 'wb') as f:
                np.save(f, feature_array)
        
        return df


    def _length_limit(self, text):
        text = str(text)
        encoded_inputs = self.tokenizer(text, max_length=511, truncation=True)
        ids = encoded_inputs["input_ids"]
        return self.tokenizer.decode(ids[1:-1])


    def _set_tokenizer(self, feature):
        model_name = self.pretrained[feature]
        self.tokenizer = BertTokenizer.from_pretrained(model_name, truncation=True, do_lower_case=True)
        return

    def _set_model(self, feature):
        model_name = self.pretrained[feature]
        self.model = BertForMultiLabelClassification.from_pretrained(model_name)
        return 

    def _set_finetuned_model(self, feature):
        model_weight_path = self.finetuned[feature]
        model_name = self.pretrained[feature]
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512, truncation=True, do_lower_case=True)
        self.model = BertRegresser.from_pretrained(model_name, config=config)
        self.model.load_state_dict(th.load(model_weight_path))
        print("load weight done!")
        return 
