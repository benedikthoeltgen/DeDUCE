
### REVISE algorithm, called by run_REVISE.py

import torch
import torch.nn as nn


    
    


def evaluate_batch(Z, model, vae, target, X_orig, lam):
    
    if torch.is_tensor(target):
        targets = target.expand((Z.shape[0],)).to(Z.device)
    else:
        targets = torch.tensor([target]).expand((Z.shape[0],)).to(Z.device)
    
    nll_error = nn.NLLLoss(reduction='none')
    l1_error = nn.L1Loss(reduction='none')
    Z.requires_grad = True
    model.zero_grad()
    vae.zero_grad()
    X = vae.decode(Z)
    y_hat = model(X)
    
    conf = y_hat[:,target].exp()
    
    loss = nll_error(y_hat, targets) + l1_error(X_orig, X).sum(dim=(1,2,3)) * lam
        
    loss.sum().backward()
    
    return conf, Z.grad



    
def generate_CE_batch(X_orig, target, model, vae, lam = .3, step_size = 1e-3, max_iter = 700, gamma = .5):
    
    model.eval() #deactivate batchnorm
    vae.eval()
    n_dim = vae.latent_dim
    Z = vae._encoder(X_orig)[:,:n_dim].detach()
    iterations = torch.ones(X_orig.shape[0]).to(X_orig.device) * max_iter
    
    for i in range(max_iter):
        
        # run model
        conf, Z_grad = evaluate_batch(Z, model, vae, target, X_orig, lam)
        
        # find mask of inputs that have not converged
        mask = (conf < gamma).view(-1, 1).repeat(1,n_dim)
        
        # record iterations needed
        just_converged = ((conf >= gamma) & (iterations == max_iter)) #and log_density > epsilon
        iterations[just_converged] = i
        
        # stopping condition
        if iterations.max() != max_iter:
            break        
        # perturb input
        Z.requires_grad = False
        Z[mask] -= Z_grad[mask] * step_size
        
        # print progress
        if i % 50 == 49 or i+1 == max_iter:
            print(f'iter {i} converged: {(iterations != max_iter).sum()}/{Z.shape[0]}')
    
    print(f'stopped after iteration {i}')
    
    X_new = vae.decode(Z).reshape(-1,28*28)
    
    # image of zeros if not converged
    mask = mask[:,0].unsqueeze(1).repeat(1,28*28)
    X_new[mask] = torch.zeros_like(X_new[mask])
    
    return X_new, iterations




