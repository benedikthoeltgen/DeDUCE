import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.latent_dim = 32
        
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, self.latent_dim*2)
        )    
        self._decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 7*7*64),
            nn.ReLU()
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.ZeroPad2d((1,0,1,0)),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def reparametrise(self, mu, logvar):
        std = logvar.exp().sqrt()
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        h = self._decoder_linear(z)
        return self._decoder(h.view((-1, 64, 7, 7)))
        
    def forward(self, x):
        z_params = self._encoder(x)
        z_mu = z_params[:,:self.latent_dim]
        z_logvar = z_params[:,self.latent_dim:]
        z = self.reparametrise(z_mu, z_logvar)
        x_recon = self.decode(z)
        return x_recon, z_mu, z_logvar, z
    
    



def VAE_loss(x, x_recon, z_mu, z_logvar, beta):
    
    # rec loss
    rec_loss = F.mse_loss(x, x_recon, reduction='sum')
    # KL divergence
    KL = - 0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    
    return rec_loss + beta * KL, rec_loss, KL
    
    
    
    
def train_VAE(model, trainloader, valloader, epochs, name, anneal=False, lr=1e-3, decay=0):
    
    cuda = torch.cuda.is_available()

    if cuda:
        print("Model on GPU")
        model.cuda()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = decay)
    
    tr_losses = torch.empty((epochs, 3))
    val_losses = torch.empty((epochs//10, 3))
    
    for epoch in range(epochs):
        
        for i, (X_batch, y_batch) in enumerate(trainloader):
            
            num_batches = 50000 / len(X_batch)
            beta = min(0.01 + (epoch * num_batches + i + 1) / (num_batches * 50) , 1) if anneal else 1.
            
            if cuda:
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                
            model.train()
            optimizer.zero_grad()
            x_recon, z_mu, z_logvar, z = model(X_batch)
            loss, rec_loss, KL = VAE_loss(X_batch, x_recon, z_mu, z_logvar, beta)
            loss.backward()
            optimizer.step()
            
            
        # trainset performance
        if epoch % 10 == 9 or epoch == 0:
            KL /= beta
            print(f'epoch {epoch+1}:   loss: {loss.item()/len(X_batch)}   rec loss: {rec_loss.item()/len(X_batch)}   KL: {KL.item()/len(X_batch)}')
        tr_losses[epoch] = torch.tensor([loss.item(), KL.item(), rec_loss.item()])
        
        
        # valset performance
        if epoch % 10 == 9:
            
            if valloader is not None:
                model.eval()
                ep_loss = 0; ep_rec_loss = 0; ep_KL = 0
                for (X_batch, y_batch) in valloader:
                    if cuda:
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()
                        
                    x_recon, z_mu, z_logvar, z = model(X_batch)
                    loss, rec_loss, KL = VAE_loss(X_batch, x_recon, z_mu, z_logvar, 1)
                    ep_loss += loss.item()
                    ep_rec_loss += rec_loss.item()
                    ep_KL += KL.item()
                
                print(f'epoch {epoch+1} val set:   loss: {ep_loss/10000}   rec loss: {ep_rec_loss/10000}   KL: {ep_KL/10000}')
                val_losses[((epoch+1)//10)-1] = torch.tensor([ep_loss, ep_rec_loss, ep_KL])
    
    
    torch.save(model.state_dict(), f'{name}_model.pt')
    torch.save(tr_losses, f'{name}_losses_tr.pt')
    torch.save(val_losses, f'{name}_losses_val.pt')
