import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super().__init__()
        self.drop_prob = 0.1
        self.input_size = 1  + 13 # [features,captions] 
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        ## LSTM layer
        self.lstm = nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=self.drop_prob,batch_first=True)
        
        ## Linear layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        embedded_captions = self.embed(captions[:,:-1])  
        x = torch.cat((features.unsqueeze(1), embedded_captions),1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        
        # return x
        return x

    def sample(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) """
		
        # output
        outputs = []
        
        for i in range(max_len+1):
            lstm_out, states = self.lstm(inputs, states)
            
            lstm_reduce = lstm_out.squeeze(1)
            output = self.fc(lstm_reduce)
            
            _, predicted_index = torch.max(output, 1)
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            if (predicted_index == 1):
                break
           
            inputs = self.embed(predicted_index)
            inputs = inputs.unsqueeze(1)
        
        return outputs
        
        