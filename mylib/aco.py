import os
import numpy as np

#A class which defines a layered graph which ants traverse.
class AntGraph:

    def __init__(self, size, initial_value=1):

        #sizes of layers of the graph
        #First layer always has only one node, from which all the ants start traversing.
        self.size = [1, *size]

        #initialize pheromone values
        self.initialize_pheromone(initial_value)


    #initialize pheromone values to the initial value
    #pheromone values are stored in a form of a list of 2D matrices since the graph is layered.
    #i-th matrix corresponds to the pheromone deposited on the edges between i-th and (i+1)th layer.
    def initialize_pheromone(self, initial_value=1):

        self.pheromone = []
        for i in range(len(self.size) - 1):
            self.pheromone.append(np.ones((self.size[i], self.size[i+1]))*initial_value)


    #Specified number of ants traverse from the first layer to the last.
    #Returns a list of 2D matrices whose elements are how many ants used the corresponding edges.
    def choose_path(self, ant_num):
        path = []

        # List of numbers of ants on each nodes.
        # The i-th element is how many ants are on the i-th node of the current layer.
        ant_nums_on_nodes = [ant_num]

        # Moves all the ants at once from a layer to the next.
        for layer in range(len(self.size) - 1):
            path.append(np.zeros((self.size[layer], self.size[layer + 1])))

            next_ant_nums_on_nodes = np.zeros(self.size[layer + 1])
            for node_i in range(self.size[layer]):
                probs = self._prob(self.pheromone[layer][node_i])
                next_ant_num = np.random.multinomial(ant_nums_on_nodes[node_i], probs)
                path[layer][node_i] = next_ant_num
                next_ant_nums_on_nodes += next_ant_num
            
            ant_nums_on_nodes = next_ant_nums_on_nodes

        return path


    #Helper function which converts a vector of pheromone values to a vector of probabilities of each edge getting chosen
    def _prob(self, vec):
        s = sum(vec)
        return [x / s for x in vec]


    #Updates pheromone values by a specified rule.
    #There are several different rules, which can be specified by update_rule = 0, 1 or 2.
    #Naive(update_rule = 0): Add reward to the selected path. How many ants selected the path is not taken into account.
    #Multiply number of ants(1): Pheromone increment is reward * (number of ants which selected the path).
    #Multiply number of succeeding paths(2): reward * (number of selected succeeding paths).
    def pheromone_update(self, path, reward, update_rule=1, evaporation_rate=0, use_limit=False, limits=[1, 20]):
        if evaporation_rate > 0:
            self._evaporate(evaporation_rate)
        
        if update_rule == 0:
            self._naive_update(path, reward)
        elif update_rule == 1:
            self._ant_num_update(path, reward)
        elif update_rule == 2:
            self._back_path_update(path, reward)

        if use_limit:
            self._clamp_pheromone(limits)
    

    #pheromone evaporation
    def _evaporate(self, evaporation_rate):
        for i in range(len(self.size) - 1):
            self.pheromone[i] *= (1 - evaporation_rate)


    #Naive way of updating pheromone.
    #Increment the pheromone values of chosen path by reward.
    #This way does not take into account how many ants chose the edge or chosen edges succeeds.
    def _naive_update(self, path, reward):
        for i in range(len(self.size) - 1):
            mask = path[i] > 0
            self.pheromone[i] += reward * mask

    
    #Updates pheromone according to how many ants chose the path.
    def _ant_num_update(self, path, reward):
        for i in range(len(self.size) - 1):
            self.pheromone[i] += reward * path[i]

    
    #Update pheromone according to how many chosen paths succeed.
    #Traverse the graph backward, just like backpropagation, to count how many chosen paths succeed.
    def _back_path_update(self, path, reward):
        #let nodes in the last layer is succeeded by one chosen path
        path_count = np.ones(self.size[-1])

        #Traverse backward
        for i in range(len(self.size)-1, 0, -1):
            #Initialize path_count for the nodes in the (i-1)th layer
            next_path_count = np.zeros(self.size[i-1])

            for j in range(self.size[i-1]):
                for k in range(self.size[i]):
                    if path[i-1][j, k] > 0:
                        self.pheromone[i-1][j, k] += reward * path_count[k]

                        next_path_count[j] += path_count[k]
            
            path_count = next_path_count

        
    #clamp pheromone values to be within a specified range
    def _clamp_pheromone(self, limits):
        for i in range(len(self.size) - 1):
            self.pheromone[i] = np.clip(self.pheromone[i], limits[0], limits[1])
        
    
    #convert ant count paths to masks which are used to dropout connections of LSTMs
    def to_mask(self, path):
        masks = []
        for i in range(len(path)):
            p = np.asarray(path[i])
            masks.append((p > 0) * 1.)

        return masks

    #percentile and mean of pheromone in each layer
    def pheromone_stat(self):
        stat = []
        for i in range(len(self.size)-1):
            p = np.array(self.pheromone[i]).reshape(-1)
            stat.append(np.percentile(p, [0, 25, 50, 75, 100]).tolist())
            stat[i].append(np.mean(p))
        return stat

    def save(self, file_dir='.', file_name='pheromone'):
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, f'{file_name}.npz')
        np.savez(file_path, *self.pheromone)


class PheromoneStatLogger:
    def __init__(self, num_graph_layers, file_dir='pheromone_stat', init_file=True, start_step=0):
        self.num_graph_layers = num_graph_layers
        self.file_dir = file_dir
        self.step = start_step

        if init_file:
            os.makedirs(self.file_dir, exist_ok=True)
            for i in range(self.num_graph_layers):
                with open(os.path.join(self.file_dir, f'layer_{i}.csv'), 'w') as f:
                    print('step,min,25-percentile,50-percentile,75-percentile,max,mean', file=f)
    
    def log_stat(self, stat, step=None):
        for i in range(self.num_graph_layers):
            with open(os.path.join(self.file_dir, f'layer_{i}.csv'), 'a') as f:
                print(f'{step if step else self.step}', end='', file=f)
                for value in stat[i]:
                    print(f',{value}', end='', file=f)
                print(file=f)
        self.step += 1
