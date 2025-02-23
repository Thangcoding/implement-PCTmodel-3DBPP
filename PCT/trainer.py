import torch 
from torch import nn 
import torch.nn.functional as F 
from model import PCT_Policy
from envs import Env
import math 
import pandas as pd
from visual import visualize

class Trainer:
    def __init__(self,step,lr,gamma,embed_dim,hidden,env,device,dataset):
        self.step = step  
        self.lr = lr 
        self.dataset = dataset       
        self.model = PCT_Policy(embed_dim= embed_dim , hidden = hidden, env= env) 
        self.gamma = gamma 
        self.optim = torch.optim.Adam(self.model.parameters(),lr = self.lr) 
        self.device = device 

    def train_step(self): 

        for i in range(len(self.dataset)): 
            items , bin_size = self.dataset['data'].loc[i], self.dataset['bin_size'].loc[i]
            items, bin_size = torch.tensor(items, dtype= torch.float).to(self.device), torch.tensor(bin_size, dtype = torch.float).to(self.device)
 
            self.logs = {'avg_actor_loss': [], 'avg_critic_loss': [],'avg_reward' : []}
            for step in range(self.step):
                self.model.train()
                self.model.env.reset()
                total_reward = 0
                total_actor_loss = 0
                total_critic_loss = 0

                for  item in items:
                    if self.model.env.internal_node == []:
                        self.model.env.bin_size = bin_size
                        self.model.env.update_state(item)
                        self.model.env.packed()
                        self.model.env.update_graph()

                        continue

                    critic_val_curr, critic_val_next ,prob , reward= self.model(item,bin_size)

                    if reward == None:
                        continue

                    reward = reward/(bin_size[0]*bin_size[1]*bin_size[2])

                    if reward != 0:
                        total_reward += reward 
                    
                        actor_loss = F.l1_loss((reward + self.gamma*critic_val_next)*torch.log(prob), critic_val_curr*torch.log(prob))
                        critic_loss = F.mse_loss(reward + self.gamma*critic_val_next,critic_val_curr)
                        total_actor_loss += actor_loss.item()
                        total_critic_loss += critic_loss.item()

                        self.optim.zero_grad()
                        (actor_loss + critic_loss).backward()
                        self.optim.step()
                # print('step ' + str(step) + ' | actor_loss: ' + str(total_actor_loss) + ' | critic_loss ' + str(total_critic_loss) + ' | reward ' + str(total_reward)) 
                self.logs['avg_actor_loss'].append(total_actor_loss) 
                self.logs['avg_critic_loss'].append(total_critic_loss) 
                self.logs['avg_reward'].append(total_reward)  
            print('Episode ' + str(i+1) + ' result: ' + ' | avg_actor_loss: '  + str(sum(self.logs['avg_actor_loss'])/len(self.logs['avg_actor_loss']))  
                                                    + ' | avg_critic_loss: ' +  str(sum(self.logs['avg_critic_loss'])/len(self.logs['avg_critic_loss'])) 
                                                    + ' | avg_reward: '      +  str(sum(self.logs['avg_reward'])/len(self.logs['avg_reward']))
                                            ) 

if __name__ == '__main__':
    df = pd.read_pickle('/Users/admin/Downloads/Machine learning/neural_CO/Bin_packing/OnlineBPP_Tree/Data/data.pkl')
    data_train = df[:1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    Learn = Trainer(step = 20,
            lr = 0.002, 
            gamma= 0.8, 
            embed_dim = 64,
            hidden = 128,
            env = Env(),
            device= device,
            dataset= data_train)

    Learn.train_step()


