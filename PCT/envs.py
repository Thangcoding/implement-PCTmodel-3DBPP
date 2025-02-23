import numpy as np 
import torch 
import torch.nn as nn 
from stable import Stable 
from visual import visualize 
from stable import Stable 

class Env:
    def __init__(self):
        self.internal_node = []
        self.leaf_node = []
        self.isvalid_leaf_node = [] 
        self.curr_item = None
        self.bin_size = None 
        self.grid_space = None 
        self.graph = {'internal_node': None,
                      'leaf_node': None, 
                      'curr_item': None, 
                      'isvalid_leaf_node': None}
    
    def update_state(self,curr_item):
        # update new item in state 
        curr_item = curr_item.tolist()
        self.graph['curr_item'] = torch.tensor([[0,0,0]+curr_item],dtype = torch.float)
        self.curr_item = curr_item 

    def grid_space_update(self, packed_item, grid_space):

        # update grid space 
        pos_x , pos_y , pos_z , x , y , z = packed_item
        pos_x , pos_y , pos_z , x , y, z = int(pos_x), int(pos_y), int(pos_z) , int(x), int(y), int(z)
        grid_space[pos_z : z + pos_z , pos_y  : y + pos_y , pos_x : x + pos_x] = torch.tensor([[[1 for _ in range(x)] for _ in range(y)] for _ in range(z)], dtype= torch.float)

        return grid_space
        

    def expand_leaf_node(self,index_leaf_node = None):
        if  index_leaf_node == None:
            EMS_space = []
            for i in range(3):
                position = [v if _ == i else 0 for _,v in enumerate(self.curr_item)]
                size = [v - self.curr_item[i] if _ == i else v for _,v in enumerate(self.bin_size)]
                space = position + size
                EMS_space.append(space)
            self.leaf_node = self.leaf_node + EMS_space 
        else:
            EMS_space = []
            leaf_node_selected = self.leaf_node[index_leaf_node]
            leaf_node_size = leaf_node_selected[3:]
            leaf_node_position = leaf_node_selected[:3]
            for i in range(3):
                position = [v + self.curr_item[i] if _ == i else v for _,v in enumerate(leaf_node_position)]
                size = [v - self.curr_item[i] if _ == i else v for _,v in enumerate(leaf_node_size)]
                space = position + size
                EMS_space.append(space)
            self.leaf_node.pop(index_leaf_node)
            self.leaf_node = self.leaf_node + EMS_space
    
    def packed(self,index_leaf_node = None):
        # implement packed item 
        if index_leaf_node == None:
            position = [0,0,0]
            packed_item = position + self.curr_item
            width, length , hight = self.bin_size.tolist()
            grid_space = torch.tensor([[[0 for _ in range(int(width))] for _ in range(int(length))] for _ in range(int(hight))])
            self.internal_node.append(packed_item)
            self.expand_leaf_node(index_leaf_node)
            self.grid_space = self.grid_space_update(packed_item, grid_space)
        else:
            leaf_node_selected = self.leaf_node[index_leaf_node]
            position = leaf_node_selected[:3]
            packed_item = position + self.curr_item 
            self.internal_node.append(packed_item)
            self.expand_leaf_node(index_leaf_node)
            self.grid_space = self.grid_space_update(packed_item, self.grid_space)

    def balance(self,leaf_node,required_area = 0.7):
        p_x, p_y, p_z = leaf_node[:3]

        if  p_z == 0:
            return True
        support_area = 0
        total_area = self.curr_item[0]*self.curr_item[1]
        required_support_area = total_area * required_area 

        for packed_item in self.internal_node:
            packed_pos, packed_size = packed_item[:3], packed_item[3:]

            if packed_pos[2] + packed_size[2] == p_z: 
                overlap_x = max(0, min(p_x + self.curr_item[0], packed_pos[0] + packed_size[0]) - max(p_x, packed_pos[0]))
                overlap_y = max(0, min(p_y + self.curr_item[1], packed_pos[1] + packed_size[1]) - max(p_y, packed_pos[1]))
                support_area += overlap_x * overlap_y

        return support_area >= required_support_area
    
    def overlap(self,leaf_node):
        item_region = []
        position = leaf_node[:3]
        size = self.curr_item 

        for packed_item in self.internal_node:
            p_x , p_y, p_z, s_x, s_y , s_z = packed_item 
            if (position[0] < p_x + s_x and position[0] + size[0] > p_x and
                position[1] < p_y + s_y and position[1] + size[1] > p_y and 
                position[2] < p_z + s_z and position[2] + size[2] > p_z):
                return True 
        return False 

    def valid_leaf_node(self):
        # check valid_leaf_node 
        isvalid = [] 
        for leaf_node in self.leaf_node: 
            print(leaf_node) 
            # if the volume of curr item greater than leaf node then it is invalid  
            if self.curr_item[0]*self.curr_item[1]*self.curr_item[2] > leaf_node[3]*leaf_node[4]*leaf_node[5]: 
                isvalid.append(0) 
            # if the size of curr item over the box  
            elif leaf_node[0] + self.curr_item[0]> self.bin_size[0] or leaf_node[1] + self.curr_item[1] > self.bin_size[1] or leaf_node[2] + self.curr_item[2] > self.bin_size[2]: 
                isvalid.append(0)   
            else: 
                if not self.overlap(leaf_node): 
                    # if self.balance(leaf_node):  
                    check = Stable(self.grid_space, self.curr_item,int(leaf_node[0]), int(leaf_node[1]), int(leaf_node[2]))
                    if check:
                        isvalid.append(1) 
                    else:
                        isvalid.append(0)
                else: 
                    isvalid.append(0)
        self.isvalid_leaf_node = isvalid
    
    def update_graph(self):
        self.graph['internal_node'] = torch.tensor(self.internal_node, dtype= torch.float)
        self.graph['leaf_node'] = torch.tensor(self.leaf_node, dtype = torch.float)
        self.graph['isvalid_leaf_node'] = torch.tensor([self.isvalid_leaf_node], dtype = torch.float)
    
    def render(self, mode = 'human'):
        # visual the environment 


        pass 
    
    def reset(self):
        self.internal_node = []
        self.leaf_node = []
        self.invalid_leaf_node = [] 
        self.curr_item = None
        self.bin_size = None 
        self.graph = {'internal_node': None,
                      'leaf_node': None, 
                      'curr_item': None, 
                      'isvalid_leaf_node': None}


if __name__ == '__main__':
    env = Env()
    env.bin_size = [100,100,100]
    list_item = [[15,17,18],[20,19,23],[18,19,20],[16,23,12]]

    for item in list_item:
        env.update_state(item)
        if env.leaf_node == []:
            env.packed()
        else:
            env.valid_leaf_node()
            valid = env.isvalid_leaf_node 
            index_valid = [i for i in range(len(valid)) if valid[i] != 0]
            if index_valid != []:
                env.packed(index_leaf_node= index_valid[0])
    check = env.internal_node + env.leaf_node[6:]

    list_items = []
    for item in check:
        list_items.append((item[:3],item[3:]))

    visualize(items=list_items, bin_size= env.bin_size)


