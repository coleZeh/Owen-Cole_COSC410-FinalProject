import torch
from torch.utils.data import DataLoader
import pandas as pd
import random
import string

class DataProcessor(torch.utils.data.Dataset):
    def __init__(self, df, max_length):
        

        self.lyrics = []
        self.popularity = []

        
        for row in range(df.shape[0]):
            self.lyrics.append(df.iloc[row]['tags_tokenized'])
            self.popularity.append(float(df.iloc[row]['track_popularity']))  
        
        #we have to make it from our data, since, no vocab file
        self.word_to_id = self.make_mapping(self.lyrics)

        self.encoded = [self.encode(seq) for seq in self.lyrics]

        self.X = [torch.tensor(seq) for seq in self.encoded]

        self.y = [torch.tensor(label) for label in self.popularity]

        self.max_length = max_length
    

    def make_mapping(self, songList):
        
        word_to_id = {'[PAD]': 0}
        startID = 1

        for song in songList:
            for word in song:
                if word not in word_to_id:
                    word_to_id[word] = startID
                    startID+=1
        
        return word_to_id
    

    def encode(self, seq):
        """Encodes sequence of words with IDs
        """
        encodedSeq = []

        for i in range(len(seq)):
            encodedSeq.append(self.word_to_id[seq[i]])
        return encodedSeq
     

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        ## Left pad
        padded_x = torch.full((1,self.max_length), self.word_to_id['[PAD]'], dtype=torch.float).flatten()
        padded_x[-x.size(0):] = x


        return padded_x, y


    def __len__(self):
        return len(self.X)

def debug():
    ## Use this function to understand how dataloader works
    pass

if __name__ == "__main__":
    debug()
