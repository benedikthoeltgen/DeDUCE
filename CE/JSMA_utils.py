
### JSMA algorithm (Papernot et al. 2016)

import torch
import torch.nn as nn

    
    
    


def evaluate_JSMA_batch(X, model, target):
    
    if torch.is_tensor(target):
        targets = target.expand((X.shape[0],)).to(X.device)
    else:
        targets = torch.tensor([target]).expand((X.shape[0],)).to(X.device)
    
    error = nn.NLLLoss(reduce=None)
    X.requires_grad = True
    model.zero_grad()
    X_reshaped = X.reshape((-1, 1, 28, 28))
    y_hat = model(X_reshaped)
    
    loss = error(y_hat, targets).mean()
    loss.backward()
    conf = y_hat[:,target].exp()
    
    return conf, loss, X.grad



    
def generate_JSMA_batch(X, labels, target, model, device, gamma=.95, delta=.2, max_iter=700, pixels = 1):
    
    X, labels = X.to(device), labels.to(device)
    model.eval()
    X = X.reshape(-1,28*28)
    iterations = torch.ones(X.shape[0]).to(device) * max_iter
    margin_mask = (torch.zeros_like(X).to(X.device) == 1)
    
    for i in range(max_iter):
        
        # run model
        conf, loss, X_grad = evaluate_JSMA_batch(X, model, target)
        
        # set grad of pixels that reached 0 or 1 to zero
        X_grad[margin_mask] = 0.0
        
        # find masks
        max_mask = X_grad.abs() == X_grad.abs().max(dim=1, keepdim=True)[0]
        for j in range(pixels-1):
            X_grad_new = torch.where(max_mask, torch.zeros_like(X_grad), X_grad) # X_grad with 0 in changed pixels
            max_mask = X_grad.abs() >= X_grad_new.abs().max(dim=1, keepdim=True)[0]
        
        conf_mask = (conf < gamma).view(-1, 1).repeat(1, max_mask.size(1))
        update_mask = (max_mask & conf_mask)
        
        # record iterations needed
        just_converged = ((conf >= gamma) & (iterations == max_iter)) #and log_density > epsilon
        iterations[just_converged] = i
        
        # stopping condition
        if iterations.max() != max_iter:
            break        
        
        # perturb input
        X.requires_grad = False
        X[update_mask] -= torch.sign(X_grad[update_mask]) * delta
        X = torch.clamp(X, 0.0, 1.0)
        
        # update mask for pixels that reached the margin
        margin_mask[(update_mask & ((X == 0) | (X == 1)))] = True
        
        # print progress
        if i % 50 == 49 or i+1 == max_iter:
            print(f'iter {i} converged: {(iterations != max_iter).sum()}/{X.shape[0]}')
    
    print(f'stopped after iteration {i}')
    
    # image of zeros if not converged
    X = X.detach()
    X[conf_mask] = torch.zeros_like(X[conf_mask])
    
    return X, iterations





