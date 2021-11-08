import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler



def get_data(data_dir, batch_size = 128, seed = None, split = 50000):
    
    if seed is not None:
            torch.manual_seed(seed)
            
    trainset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    
    indices = list(range(len(trainset)))
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, sampler=train_sampler)
    valloader = torch.utils.data.DataLoader(trainset, batch_size, sampler=val_sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size)
        
    return trainloader, valloader, testloader


def get_fashion(data_dir, batch_size = 128, seed = None):
    
    if seed is not None:
            torch.manual_seed(seed)
            
    dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)
        
    return dataloader






def train(model, args, trainloader, valloader=None):
    
    epochs = args.epochs
    error = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25,40], gamma=0.1)
    
    for epoch in range(epochs):
        correct = 0
        model.train()
        
        for i, (X_batch, y_batch) in enumerate(trainloader):
            
            if args.cuda:
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                
            optimizer.zero_grad()
            output = model(X_batch)
            loss = error(output, y_batch)
            loss.backward()
            optimizer.step()
            
            # Total correct predictions this epoch
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == y_batch).sum()
            
            if epochs == 1:
                print(f'batch {i+1}/{len(trainloader)} done')
        
        # trainset performance
        train_acc = round(float(correct*100)/float(50000),2)
        print(f'epoch {epoch+1}:   train_acc: {train_acc}%')
        
        scheduler.step()
        
        
        
        if (epoch+1) % 5 == 0:
            
            # valset performance
            if valloader is not None:
                model.eval()
                correct = 0
                for (X_batch, y_batch) in valloader:
                    if args.cuda:
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()
                    output = model(X_batch)
                    predicted = torch.max(output.data, 1)[1] 
                    correct += (predicted == y_batch).sum()
                val_acc = round(float(correct*100)/float(10000),2)
                print(f'validation set accuracy: {val_acc}%')
                
            torch.save(model.state_dict(), args.out_dir + args.model_name + f'_e{epoch}.pt')
        
    return model
        