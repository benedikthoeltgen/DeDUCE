
### main algorithm to generate counterfactuals
### first half: efficient algorithm for batches (used in run_CE_alg.py)
### second half: algorithm for single inputs for exploring (used in slow_CE_generation.py)


import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

    
    
    


def evaluate_batch(X, model, target, gmm, loss_coeff, lossgrad):
    
    if torch.is_tensor(target):
        targets = target.expand((X.shape[0],)).to(X.device)
    else:
        targets = torch.tensor([target]).expand((X.shape[0],)).to(X.device)
    
    error = nn.NLLLoss(reduction='none')
    X.requires_grad = True
    model.zero_grad()
    X_reshaped = X.reshape((-1, 1, 28, 28))
    y_hat = model(X_reshaped)
    features = model.feature
    
    conf = y_hat[:,target].exp()
    log_probs = gmm.log_prob(features[:, None, :])
    log_dens_target = error(log_probs, targets)
    log_density = torch.logsumexp(log_probs, 1)
    
    if lossgrad:
        classif_loss = error(y_hat, targets)
        loss = log_dens_target + classif_loss * loss_coeff
        
    elif loss_coeff != 0:
        alpha = log_dens_target.clone().detach().reciprocal().abs()
        classif_loss = error(y_hat, targets)
        beta = classif_loss.clone().detach().reciprocal()
        loss = log_dens_target * alpha + classif_loss * beta * loss_coeff
        
    else: 
        loss = log_dens_target
        
    loss.sum().backward()
    
    return conf, log_dens_target, log_density, X.grad



    
def generate_CE_batch(X, labels, target, model, gmm, gamma=.5, delta=.2, max_changes=5, max_iter=700, 
                      momentum = .6, loss_coeff = 1, pixels = 1, lossgrad = False):
    
    X, labels = X.to(gmm.loc.device), labels.to(gmm.loc.device)
    model.eval() #deactivate batchnorm
    X = X.reshape(-1,28*28)
    pixel_changes = torch.zeros_like(X).to(X.device)
    X_grad_last = torch.zeros_like(X).to(X.device) #for momentum
    iterations = torch.ones(X.shape[0]).to(X.device) * max_iter
    
    for i in range(max_iter):
        
        # run model
        conf, log_dens_target, log_density, X_grad = evaluate_batch(X, model, target, gmm, loss_coeff, lossgrad)
        
        # set grad of pixels with max changes to zero
        X_grad[pixel_changes >= max_changes] = 0.0
        
        # find masks
        max_mask = X_grad.abs() == X_grad.abs().max(dim=1, keepdim=True)[0]
        for j in range(pixels-1):# only called if pixels > 1 since range(0) gives empty list
            X_grad_new = torch.where(max_mask, torch.zeros_like(X_grad), X_grad) # X_grad with 0 in changed pixels
            max_mask = X_grad.abs() >= X_grad_new.abs().max(dim=1, keepdim=True)[0]
        conf_mask = (conf < gamma).view(-1, 1).repeat(1, max_mask.size(1))
        update_mask = (max_mask & conf_mask)
        
        # record iterations needed
        just_converged = ((conf >= gamma) & (iterations == max_iter))
        iterations[just_converged] = i
        
        # stopping condition
        if iterations.max() != max_iter:
            break        
        
        # add momentum
        X_grad += X_grad_last * momentum
        
        # perturb input
        X.requires_grad = False
        X[update_mask] -= torch.sign(X_grad[update_mask]) * delta
        X = torch.clamp(X, 0.0, 1.0)
        
        # update variables
        X_grad_last = X_grad
        pixel_changes[update_mask] += 1.
        
        # print progress
        if i % 50 == 49 or i+1 == max_iter:
            print(f'iter {i} converged: {(iterations != max_iter).sum()}/{X.shape[0]}')
    
    print(f'stopped after iteration {i}')
    
    # image of zeros if not converged
    X = X.detach()
    X[conf_mask] = torch.zeros_like(X[conf_mask])
    
    return X, iterations, pixel_changes










#####################################################################################################
#####     for single CEs
#####################################################################################################



def evaluate_input(x, model, target, gmm, loss_coeff):
    
    error = nn.NLLLoss()
    x.requires_grad = True
    model.zero_grad()
    assert x.grad is None
    y_hat = model(x)
    features = model.feature
    
    conf = y_hat[0][target].exp().item()
    log_probs = gmm.log_prob(features[:, None, :])
    log_dens_target = log_probs[0][target]
    log_density = torch.logsumexp(log_probs[0], 0)
    
    loss = error(y_hat, target)
    
    combined_loss = loss / loss.abs().item() * loss_coeff - log_dens_target / log_dens_target.abs().item()
    
    return combined_loss, conf, log_dens_target.item(), log_density.item(), loss.item(), features



def generate_CE(x, label, target, model, gmm, gamma=.95, delta=.2, 
                max_changes=5, max_iter=1000, momentum = .5, loss_coeff = 0, pixels = 1, enable_plots=True):
    
    model.eval()
    P = torch.zeros([28*28])
    c = 0
    x_grad_masked_last = torch.zeros([28*28])
    target = target if torch.is_tensor(target) else torch.tensor([target])
    
    # run model on initial input
    combined_loss, conf, log_dens_target, log_density, loss, feat_orig = evaluate_input(x, model, target, gmm, loss_coeff)
    
    while(conf < gamma and c < max_iter):
        
        # find feature to perturb
        combined_loss.backward()
        x_grad_masked = torch.where(P < max_changes, x.grad.view(28*28) + momentum*x_grad_masked_last, torch.zeros_like(P))
        _, i = torch.max(torch.abs(x_grad_masked), 0) #P[i] must be < max_changes
        
        # find second feature to perturb
        x_grad_masked2 = x_grad_masked
        x_grad_masked2[i] = 0
        _, j = torch.max(torch.abs(x_grad_masked2), 0)
        
        # find third feature to perturb
        x_grad_masked3 = x_grad_masked2
        x_grad_masked3[j] = 0
        _, k = torch.max(torch.abs(x_grad_masked3), 0)
        
        # find fourth feature to perturb
        x_grad_masked4 = x_grad_masked3
        x_grad_masked4[k] = 0
        _, l = torch.max(torch.abs(x_grad_masked4), 0)
        
        # create perturbed input
        x.requires_grad = False
        x.view(28*28)[i] -= torch.sign(x.grad.view(28*28)[i]) * delta
        if pixels > 1:
            x.view(28*28)[j] -= torch.sign(x.grad.view(28*28)[j]) * delta
            if pixels > 2:
                x.view(28*28)[k] -= torch.sign(x.grad.view(28*28)[k]) * delta
                if pixels > 3:
                    x.view(28*28)[l] -= torch.sign(x.grad.view(28*28)[l]) * delta
        x = torch.clamp(x, 0, 1)
        
        # run model on new input
        combined_loss, conf, log_dens_target, log_density, loss, feat = evaluate_input(x, model, target, label, gmm)
        
        # update variables
        x_grad_masked_last = x_grad_masked
        P[i] += 1
        c += 1
        
        # print progress
        if c % 20 == 0 or c == max_iter:
            print(f'iter {c}  conf: {round(conf, 4)}   log_dens_target: {log_dens_target}')
            print(f'          loss: {round(loss, 4)}   log_density: {log_density}')
            
            if enable_plots:
                x_new = x
                plt.axis('off')
                plt.imshow(x_new.detach().numpy()[0][0], cmap='gray')
                plt.show()
    
    print(f'stopped after iteration {c}')
    if c % 20 != 0 and c != max_iter:
        print(f'conf: {round(conf, 4)}   log_dens_target: {log_dens_target}')
        print(f'loss: {round(loss, 4)}   log_density: {log_density}')
        
    return x, c, P





def save_fig(data, name):
    new_data = np.zeros(np.array(data.shape) * 10)
    
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            new_data[j * 10: (j+1) * 10, k * 10: (k+1) * 10] = data[j, k]
    
    plt.imsave(name + '.png', new_data)
    
    
    
    
    
    