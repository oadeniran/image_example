# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:29:05 2022

@author: DAVID
"""

import torch
import random
import numpy as np
from collections import deque
from ICA_Snake_Game import AISnakeGame, Direction, Point
from Model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE =  1000
LR = 0.001
class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon =0 # control randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY)  
        self.model = Linear_QNet(11, 256, 3)
        self.trainer =QTrainer()
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y +20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        #Danger straight
        
        state = [
            (dir_r and game.is_collision(point_r))or
            (dir_l and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),
            
            #danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_l and game.is_collision(point_u))or
            (dir_r and game.is_collision(point_d)),
            
            #danger left
            (dir_d and game.is_collision(point_r))or
            (dir_u and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),
            
            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #location of food
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, # food is rigt
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y] # food is down
            
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # pop lef if max memory is reached
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample =random.sample(self.memory, BATCH_SIZE) # return list of tuples
        else:
            mini_sample = self.memory
            states, actions, rewards, next_states, game_overs = zip(mini_sample)
            
            self.trainer.train_step(states, actions, rewards, next_states, game_overs)
         
    def train_short_memory(self, state, action, reward, next_state, game_over):
            self.trainer.train_step(state, action, reward, next_state, game_over)
   
    def get_action(self, state):
        
        #random moves: tradeoff betwwen exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move =[0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) 
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() 
            final_move[move]= 1
            
        return final_move
            
def train():
        plot_scores = []
        plot_mean_scores = 0
        total_scores = 0
        record = 0
        agent= Agent()
        game = AISnakeGame()
        while True:
            #get the old state 
            state_old = agent.get_state(game)
            
            # getmove
            final_move = agent.get_action(state_old)
            
            #perform move and get new state
            reward, game_over, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            #train agent short memory for one step only
            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
            #remember
            agent.remember(state_old, final_move, reward, state_new, game_over)
            
            if game_over:
                #train long memory or replay memory
                #plot the result
                game.reset()
                agent.n_games +=1
                agent.train_long_memory()
                
            if score > record:
                    record = score
                    agent.model.save()
                    
                    
                    print('Game', agent.n_games, 'Score', score, 'Record:', record)
                    
                    plot_scores.append(score)
                    total_scores += score
                    mean_scores = total_scores/agent.n_games
                    plot_mean_scores.append(mean_scores)
                    plot(plot_scores, plot_mean_scores)
                    
        
    
if __name__ =='__main__':
        train()