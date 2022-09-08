import torch
import torch.nn as nn
import numpy as np
import os
import random
from transformers import AutoConfig, AutoModel, AutoTokenizer


device = torch.device('cpu')


labels = {
 0: 'toxic',
 1: 'severe_toxic',
 2: 'obscene',
 3: 'threat',
 4: 'insult',
 5: 'identity_hate',
 }
 
MODEL_NAME='roberta-base'
NUM_CLASSES=6
 
MAX_LEN = 128
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
 
class ToxicModel(torch.nn.Module):
    def __init__(self):
        super(ToxicModel, self).__init__()
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(MODEL_NAME)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": NUM_CLASSES,
            }
        )
        self.transformer = AutoModel.from_pretrained(MODEL_NAME, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, NUM_CLASSES)  
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        transformer_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = transformer_out[0]
        sequence_output = self.dropout(torch.mean(sequence_output, 1))
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return logits
 
 
def inference_fn(model, input_ids=None, attention_mask=None, token_type_ids=None):  
    model.eval()
    input_ids = input_ids[0].to(device)  
    attention_mask = attention_mask[0].to(device)  
    token_type_ids = token_type_ids[0].to(device)  
    
    with torch.no_grad():
        output = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), token_type_ids.unsqueeze(0))
    out = output.sigmoid().detach().cpu().numpy().flatten()
       
    return out
   
def predict(comment=None) -> dict:  
    text = str(comment)
    text = " ".join(text.split())

    inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]
    
    ids = torch.tensor(ids, dtype=torch.long),
    mask = torch.tensor(mask, dtype=torch.long),
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long),
    
    model = ToxicModel()
    
    model.load_state_dict(torch.load("toxicx_model_0.pth", map_location=torch.device(device)))
    model.to(device)

    predicted = inference_fn(model, ids, mask, token_type_ids)
  
    return {labels[i]: float(predicted[i]) for i in range(NUM_CLASSES)}
    

