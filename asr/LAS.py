#!/usr/bin/env python
# coding: utf-8

# In[79]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#We should notice that the input size regulation in the LSTM is (seq_len,batch,input_dim)
#When we turn on the batc_first,input and output will become (batch,seq_len,input_dim)
#Also,the input argument in the LSTM(*args,**kwargs) , so we can input the list!!

class TimeDistributed(nn.Module):
    def __init__(self,module,batch_first=True):
        super(self,TimeDistributed).__init__()
        self.module = module
        self.batch_first = batch_first
    def forward(self,input_seq):
        assert len(input_seq.size())>2
        ##reshape input shape  (samples,timestep,input_size) --> (timestep*sample,input_size)
        reshaped_input = input_seq.contiguous().view(-1,input_seq.size(-1))
        output = self.module(reshaped_input)
        #with the input --> (batch==samples,seq_len==timestep,input_dim)
        if self.batch_first: 
            #(samples,timestep,input_size)
            output = output.contiguous().view(input_seq.size(0),-1,output.size(-1))
        else:
            #(timestep,samples,input_size)
            output = output.contiguous().view(-1,input_seq.size(-1),output.size(-1))
        
        return output
            
#Do the resolution 
class PBLSTM(nn.Module):
    def __init__ (self,input_dim,hidden_size,batch_first,rnn_unit = 'LSTM',dropout_rate= 0.0):
        super(PBLSTM,self).__init__()
        self.rnn = getattr(nn,rnn_unit.upper()) #將nn.LSTM call 出來
        self.BLSTM = self.rnn(input_dim,hidden_size,1,bidirectional=True,dropout=dropout_rate,
                              batch_first=True)
    def forward(self,input_x):
        batch_size = input_x(0)
        timestep = input_x(1)
        feature_dim = input_x(2)
        ##減少timestep的數量、增加feature dimensions
        input_x = inpiut_x.contiguous().view(batch_size,(timestep/2),feature_dim*2)
        output,hidden = self.BLSTM(input_x)

#INPUT:sequence (batch_size,timestep,input_feature_dim)
#Every unit is pBLSTM and did the resolution so that the parameter can be reduced.
class Listener(nn.Module):
    def __init__(self,input_feature_dim,hidden_size,listener_layers,rnn_unit,dropout_rate=0.0):
        super(Listener,self).__init__()
        self.listener_layers = listener_layers
        self.PBLSTMlayer0 = PBLSTM(input_feature_dim,hidden_size,rnn_unit=rnn_unit,dropout=dropout_rate)
        
        for i in range(1,self.listener_layers):
            setattr(self,'PBLSTMlayer'+str(i),PBLSTM(hidden_size*2,hidden_size,rnn_unit=rnn_unit,
                                                     dropout=dropout_rate))
        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()
    def forward(self,input_x):
        output,_ = self.PBLSTMlayer0(input_x)
        for i in range(1,self.listenr_layers):
            output,_ = getattr(self,'PBLSTM'+str(i))(output)
        return output


class Speller(nn.Module):
    def __init__(self,output_class_dim,speller_hidden_dim,rnn_unit,speller_rnn_layer,use_gpu,max_label_len,
                 use_mlp_in_attention, mlp_dim_in_attention, mlp_activate_in_attention, listener_hidden_dim,
                 multi_head, decode_mode, **kwargs):
        super(self,Speller).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.label_dim = output_class_dim
        self.gpu  = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        
        self.rnn_layer = self.rnn_unit(output_class_dim+speller_hidden_dim,speller_hidden_dim,
                                       speller_rnn_layer,batch_first=True)
        self.attention = Attention(preprocess_input=use_mlp_in_attention,preprocess_dim=mlp_dim_in_attention
                                   ,activate=mlp_activate_in_attention,input_feature_dim=2*listener_hidden_dim
                                   ,head_number=multi_head)
        self.char_distribution = nn.Linear(speller_hidden_dim*2,output_class_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        if self.gpu:
            self = self.cuda()
    #decoder state是具有 speller_hidden_dim的另外一種hidden_featurre
    #得到的方式為，一個initial rnn_input 是concat (1)全零的output sample
    #                                          (2)listener_feature的一個timestep的feature
    #rnn_input 得到後，丟進rnn_layer後得到的即為 decoder_state:(batch,1,speller_hidden_dim) == rnn_output
    #rnn_output再和listener_feature做attention後得到context vector
    #Concat context_vector and decoder_state and put into nn.Softmax(nn.Linear
    #                                                                (2*speller_dim,output_class_dim)) == y1
    
    def forward_step(self,initial_word,last_hidden_state,listener_feature):
        rnn_output,rnn_hidden = self.rnn_layer(initial_word,last_hidden_state)
        attention_score,context_vector = self.attention(rnn_output,listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1),context_vector],dim=-1)
        raw_predict = self.softmax(self.char_distribution(concat_feature))
        
        return raw_predict,rnn_hidden,attention_score,context_vector
    
    def forward(self,listener_feature,ground_truth=None,teacher_force_rate=0.9):
        if ground_truth==None:
            teacher_force_rate = 0
        batch_size = listener_feature.size()[0]
        output_word = nn.functional.one_hot(self.float_type(np.zeros((batch_size,1))),self.label_dim)
        
        if self.use_gpu:
            output_word = output_word.cuda()
        
        rnn_input = torch.cat([output_word,listener_feature[:,0:1,:]],dim=-1)

        hidden_state = None
        raw_pred_seq = []
        output_seq = []
        attention_record = []
        
        if (ground_truth is None) or (not teacher_force):
            max_step = self.max_label_len
        else:
            max_step = ground_truth.size()[1]
        #若不是teacher forcing，則我們rnn_input = concat(context,)
        #若是teacher forcing，則我們rnn_input = concat(context,real_label_vector)
        for step in range(max_step):
            raw_predict,rnn_hidden,attention_score,context = self.forward_step(rnn_input,hidden_state
                                                                                     ,listener_feature)
            raw_pred_seq.append(raw_predict)
            attention_record.append(attention_score)
            
            if teacher_force:
                output_word = ground_truth[:,step:step+1,:].type(self.float_type)
            else:
                # Case 0. raw output as input
                if self.decode_mode == 0:
                    output_word = raw_pred.unsqueeze(1)
                # Case 1. Pick character with max probability
                elif self.decode_mode == 1:
                    output_word = torch.zeros_like(raw_pred)
                    for idx,i in enumerate(raw_pred.topk(1)[1]):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)             
                # Case 2. Sample categotical label from raw prediction
                else:
                    sampled_word = Categorical(raw_pred).sample()
                    output_word = torch.zeros_like(raw_pred)
                    for idx,i in enumerate(sampled_word):
                        output_word[idx,int(i)] = 1
                    output_word = output_word.unsqueeze(1)
                
            rnn_input = torch.cat([output_word,context.unsqueeze(1)],dim=-1)

        return raw_pred_seq,attention_record

            
        
    
#INPUT: (1)Listener_feature = [batch_size,timestep,listener_feature_dim]
#       (2)Decoder_state = [batch_size,1,decoder_hidden_dim]
#OUTPUT: (1)attention_weight = [batch,timestep]
#        (2)context_vector = [batch,listner_feature_dim]
class Attention(nn.Module):
    def __init__(self,preprocess_input,preprocess_dim,activate,mode='dot',
                 input_feature_dim=512,head_number=1,):
        self.mode = mode.upper()
        self.preprocess_input = preprocess_input
        self.head_number = head_number
        self.softmax = nn.Softmax(dim=-1)
        if preprocess_input:
            self.preprocess_dim = preprocess_dim
            self.mh = nn.Linear(input_feature_dim,preprocess_dim*head_number)
            self.sh = nn.Linear(input_feature_dim,preprocess_dim)
            if self.head_number>1:
                self.dimreduction = nn.Linear(input_feature_dim*head_number,input_feature_dim)
            if activate != None:
                self.activate = getattr(F,activate)
            else:
                self.activate = None
    def forward(self,decoder_state,listener_feature):
        if self.preprocess_input:
            if self.activate:
                comp_decoder_state = self.activate(self.mh(decoder_state))
                comp_listener_feature = self.activate(TimeDistributed(self.sh,listener_feature))
            else:
                comp_decoder_state = self.mh(decoder_state)
                comp_listener_feature = TimeDistributed(self.sh,listener_feature)
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature
            
        if self.mode == 'dot':
            if self.head_number == 1:
                #Use the matrix_multiply (1,decoder_hidden_state)*(listener_feature_dim,timestep)
                #After squeeze: (batch,1,timestep) --> (batch,timestep)
                energy = torch.bmm(decoder_state,listener_feature.transpose(1,2)).squeeze(dim=1)
                attention_score = [self.softmax(energy)] # size:(batch,timestep)
                
                #(batch,timestpe,feature_dim)*(1,timestep)
                context_vec = torch.sum(listener_feature*attention_score[0].unsqueeze(2)
                                        ,dim=1)
            ##multi_head Right Now!!
            else:
                
                context_vec = self.dim_reduce(torch.cat(projected_src,dim=-1))
        elif self.mode == 'additive':
            pass
        
        return attention_score,context_vec
        

