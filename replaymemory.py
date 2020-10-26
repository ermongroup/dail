import random
from collections import deque

# Initialize replay buffers
def create_replay_memory(env, params):
    '''
    Creates separate replay memories for each domain
    Args:
        env : environments for each domain : dict
        params : experiment parameters : dict
        memtype : type of replay memory to create : str
    Returns:
        replay_memory : replay memories for each domain : dict
    '''

    replay_memory = {}
    for d_ in env.keys():
        if params['train']['memtype'] == 'vanilla':
            replay_memory[d_] = FIFOdeque(memsize=params['train']['memsize'])
        else:
            print("[replaymemory.py] Unrecognized replay buffer type: {}".format(memtype))

    replay_memory['model'] = FIFOdeque(memsize=params['train']['memsize'])

    return replay_memory

class FIFOdeque():
    '''
    Vanilla replay memory with FIFO structure implemented with a deque
    '''
    def __init__(self, memsize):
        self.memory = deque(maxlen=memsize)

    def len(self):
        '''
        Number of entries in the replay memory
        Args:
        Returns:
            len : number of entries in the memory : int
        '''
        return len(self.memory)

    def add_to_memory(self, data):
        '''
        Add data to the end of the deque
        Args:
            data : data entry to add : tuple
        Returns:
            None
        '''
        self.memory.append(data)

    def sample_from_memory(self, batchsize):
        '''
        Sample a batch from memory
        Args:
            batchsize : size of the batch to sample : int
        Returns:
            sample : sampled batch : list of
        '''
        sample = random.sample(self.memory, batchsize)
        return sample

    def set_memory(self, loaded_deque):
        '''
        Sample a batch from memory
        Args:
            loaded_deque : deque loaded from a saved file
        Returns:
            None
        '''
        self.memory = loaded_deque









