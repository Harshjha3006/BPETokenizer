from baseClass import get_stats,merge,Base
import typer
from typing import List

app = typer.Typer()

class BasicTokeniser(Base):
    def __init__(self):
        super().__init__()

    def train(self,text,vocab_size):
        
        ids = list(text.encode(encoding = "utf-8"))

        num_merges = vocab_size - 256
        vocab = {id : bytes([id]) for id in range(256)}
        merges = {}
        for i in range(num_merges):
            stats = get_stats(ids)
            maxPair = max(stats,key = stats.get)
            newId = 256 + i
            ids = merge(ids,maxPair,newId)
            merges[maxPair] = newId
            vocab[newId] = vocab[maxPair[0]] + vocab[maxPair[1]]

        self.merges = merges
        self.vocab = vocab

    def encode(self,text):

        ids = list(text.encode(encoding = "utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)
            minPair = min(stats,key = lambda p : self.merges.get(p,float('inf')))

            if minPair not in self.merges:
                break

            ids = merge(ids,minPair,self.merges[minPair])

        return ids

    def decode(self,ids):

        byteStr = b"".join(self.vocab[id] for id in ids)
        return byteStr.decode(encoding = "utf-8",errors = "replace")




@app.command()
def train(file : str,vocab_size :int,model_file_prefix : str):
    tokeniser = BasicTokeniser()
    # convert file to text string
    text = ""
    with open(file,'r') as f:
        text = f.read().replace("\n"," ")
    
    #train the tokeniser on the text file
    tokeniser.train(text,vocab_size)
    #save the merges dictionary in a file
    tokeniser.save_merges(model_file_prefix)

@app.command()
def encode(text : str,merges_file : str):
    tokeniser = BasicTokeniser()
    tokeniser.load_merges(merges_file)
    #encode the given text using the merges_file
    encoding = tokeniser.encode(text)
    print(encoding)

@app.command()
def decode(ids : List[int],merges_file : str):

    tokeniser = BasicTokeniser()
    tokeniser.load_merges(merges_file)
    #decode the ids using th vocab file
    text = tokeniser.decode(ids)
    print(text)


if __name__ == "__main__":
    app()