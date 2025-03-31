import torch
from sklearn.metrics import mean_squared_error
#I BASICALLY USED THIS FROM OUR LAB 6
def batch_train(model, train_loader, valid_loader, num_epochs, lr=0.001):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for X,y in train_loader:
            ## Complete this loop
            ## Hint: You might have to reshape y_pred and flatten y. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                ## From pytorch: This criterion computes the cross entropy loss between input logits (y_pred) and target (y)
                    ## Input: Shape (N,C) where N=batchSize*sentenceLength and C=vocabSize
                    ## Target: Shape (N)
            #print(X.size(0))
            hidden = model.init_hidden(15000)
            y_pred, hidden = model(X, hidden)
            y_pred = y_pred.view(-1) 
            y = y.view(-1).float() 

            loss = model.loss(y_pred,y)
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) #helps with exploding gradient
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        val_loss = round(compute_loss(model, valid_loader).item(), 5)#Round does not work on tensor's; thus, we use .item().
        if epoch%5 == 0:
            print(f"Epoch {epoch+1}:\t Avg Train Loss: {round(epoch_loss/num_batches,5)}\t Avg Val Loss: {val_loss}")



##I USED THIS FROM OUR LAB 6
@torch.no_grad()  ## ensures gradient not computed for this function. 
def compute_loss(model, data_loader):
    """
    Returns avg loss of the model on the data
    """
    total_loss = 0
    hidden = model.init_hidden(15000)
    for i, datapoint in enumerate(data_loader):
        ## Implement this loop
        X= datapoint[0]
        y= datapoint[1]
        
        #if i==0:#we only need to initilize the hidden layer once when evaluating. We do not need to zero the layer after each sentence because prior sentences can help the model make predicitons in subsequent sentences.
            #hidden = model.init_hidden(X.size(0))
        
        y_pred, hidden = model(X, hidden)
        y_pred = y_pred.view(-1).float()
        y = y.view(-1).float()
        loss = model.loss(y_pred,y)
        total_loss += loss

    return total_loss/(i+1)

@torch.no_grad()
def evaluate(model, test_loader):
    """
    Function to evaluate the recall, precision, and accuracy of the model
    """
    model.eval()
    pred = []
    true = []
    # initialize the hidden state once as it should carryover
    hidden = model.init_hidden(15000) 
    for X, y in test_loader:

        y_pred, hidden = model(X, hidden)

        # get the predictions and true labels
        pred.extend(y_pred.cpu().numpy())
        true.extend(y.cpu().numpy())

    print(pred)
    print(true)
    mse = mean_squared_error(pred, true)
    return mse