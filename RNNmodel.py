import torch
class RNNmodel(torch.nn.Module):
    def __init__(self, vocabSize, nEmbed, nHidden, nLayers):
        super(RNNmodel, self).__init__()

        self.nLayers=nLayers
        self.nHidden=nHidden
        ## Implement this. You  must use.
        # 1. Embedding layer
        # 2. RNN layer
        # 3. A final linear layer to create a classification prediction
        # 4. a sigmoid function to generate a prediction percentage
        #self.dropout = torch.nn.Dropout(0.50)
        self.embeddingLayer = torch.nn.Embedding(vocabSize, nEmbed)
        self.RNNLayer = torch.nn.RNN(nEmbed, nHidden, nLayers)
        self.regression = torch.nn.Linear(nHidden, 1)

    def forward(self, X, hidden):
        ## Forward pass returns prediction and hidden
        #embedds the sentences
        embedded = self.embeddingLayer(X.long())
        #gets the hidden and output to pass
        output, hidden = self.RNNLayer(embedded, hidden)
        #only want to pass the last timestep (end of sentence) to the classification layer
        output = output[:, -1, :] 
        output = self.regression(output)
        return output, hidden
        
    def init_hidden(self, batchSize):
        return torch.zeros((self.nLayers, batchSize, self.nHidden), dtype=torch.float)
        

    def loss(self, y_hat, y):
        ## Using pytorch binary cross-entropy loss
        loss = torch.nn.MSELoss()
        return loss(y_hat, y)
