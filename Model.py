# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:41:29 2022

@author: DAVID
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
    
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.linear1 =nn.Linear (input_size, hidden_size)
            self.linear2 =nn.Linear(hidden_size, output_size)
   
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x =self.linear2(x)
        return x
    
    
    def save(self, file_name ='model.pth'):
        model_folder_path= './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_name)
            
            
class QTrainer:
    
    def __int__(self, Model, lr, gamma, train_step):
            #Learning rate
            self.lr =lr
            Discount 
            self.gamma = gamma
            self.Model = Model
        
            self.optimi = optim.Adam(Model.parameters(), lr = self.lr)
            self.criterion = nn.MSELoss()
            
    def train_step(self, state, action, reward, next_state, game_over):
        state =torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype =torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch .tensor(reward, dtype=torch.float)
        #(n, x)
        
        if len(state.shape)==1:
            #(1,x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
            
            #1: get predicted Q values with current state
            prediction = self.Model(state)
            
            target = prediction.clone()
            for index in range(len(game_over)):
                Q_new =reward[index]
                if not game_over[index]:
                    Q_new = reward[index] + self.gamma * torch.max(self.Model(next_state[index]))
            
            
            target[index][torch.argmax(action).item()] = Q_new
            #2: Q_new =reward + gamma *maximum(nxt predicted Q value)
            #prediction.clone()
            
            self.optimizer.zero_grad()
            loss = self.criterion(target, prediction)
            loss.backward()
            
            self.optimizer.step()
            
            #prediction[argmax(action)] = Q new value
            