# Helper functions 


def get_stats(ids,stats = None):

    """
    returns a dictionary containing a frequency of all consecutive pairs in a list of token ids
    """

    counter = {} if stats == None else stats
    for i in range(len(ids) - 1):
        pair = (ids[i],ids[i + 1])
        counter[pair] = counter.get(pair,0) + 1
    return counter


def merge(ids,pair,newId):

    """
    merges the given pair into a new token
    """
    newIds = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and pair[0] == ids[i] and pair[1] == ids[i + 1]:
            newIds.append(newId)
            i += 2
        else:
            newIds.append(ids[i])
            i += 1
    return newIds


# Base Class for a tokeniser


class Base:
    def __init__(self):
        self.merges = {} # a dict of (tokenId1,tokenId2) -> newTokenId
        self.vocab = {} # a dict of (tokenId) -> byte representation of the token
    
    def train(self,text,vocab_size):
        raise NotImplementedError
    
    def encode(self,text):
        raise NotImplementedError
    
    def decode(self,ids):
        raise NotImplementedError
    
    def __build_vocab(self):
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save_merges(self,file_prefix):
        file = file_prefix + '.merges'
        with open(file,'w') as f:
            f.write("merges\n")
            for p0,p1 in self.merges:
                f.write(f"{p0} {p1}\n")
        
    def load_merges(self,file):
        assert file.endswith(".merges")
        merges = {}
        idx = 256

        with open(file) as f:
            header = f.readline().strip()
            assert header == "merges"
            for line in f:
                idx1,idx2 = map(int,line.split())
                merges[(idx1,idx2)] = idx
                idx += 1
        
        self.merges = merges
        self.vocab = self.__build_vocab()