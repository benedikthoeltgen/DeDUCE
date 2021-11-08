import torch
import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, z_dim=64, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Sigmoid(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)



class Discriminator(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(im_chan, hidden_dim, 5, 2, padding=2, bias=False)),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_dim, hidden_dim * 2, 5, 2, padding=2, bias=False)),
        )
        self.disc2 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_dim * 2, 1, 5, 2, bias=False)),
        )

    def feature(self, image):
        return self.disc1(image)

    def forward(self, image):
        disc_pred = self.disc2(self.disc1(image))
        return disc_pred.view(len(disc_pred), -1)




def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)




def calc_AGAN(images, steps=4000, lam=.1, step_size=0.03, mom = .0, rep=3, device='cuda'):
    
    num_steps = 380
    name = 'WGAN41m'
    
    gen = Generator().to(device) 
    disc = Discriminator().to(device) 
    gen.load_state_dict(torch.load(f'./metrics/{name}_gen_{num_steps}.pt', map_location=device))
    disc.load_state_dict(torch.load(f'./metrics/{name}_disc_{num_steps}.pt', map_location=device))
    gen.eval()
    disc.eval()

    im_features = disc.feature(images.view(len(images),1,28,28))
    #if random:
    #    z = get_noise(len(images), 64, device=images.device)
    #else:
    loss_min = torch.ones(len(images),) * float('inf')
    stepsize = step_size
    #for stepsize in [step_size, step_size*5]:
    for i in range(rep):
        #z = torch.zeros((len(images),64), device=images.device)
        z = get_noise(len(images), 64, device=images.device)
        z_grad = torch.zeros_like(z).to(images.device)
        for i in range(steps):
            z.requires_grad = True
            z.grad = None
            disc.zero_grad()
            gen.zero_grad()
            assert z.grad is None
            new_images = gen(z)
            res_loss = torch.abs(images - new_images).sum((1,2,3))
            disc_loss = torch.abs(im_features - disc.feature(new_images)).sum((1,2,3))
            loss = res_loss + disc_loss * lam
            loss_reduced = loss.sum()
            loss_reduced.backward(retain_graph=True)
            z.requires_grad = False
            #z_grad = z.grad.clone()
            z -= (z.grad + mom*z_grad) * stepsize
            z_grad = z.grad

            if i == steps - (steps // 5):
                stepsize /= 10
        loss_min = torch.min(loss_min, loss.detach().cpu())
        
    return loss_min.numpy()




def calc_AGAN_dm(images, steps=2000, lam=.2, step_size=0.05, random=False, device='cuda'):
    
    num_steps = 300
    name = 'WGAN51dm'
    
    gen = Generator().to(device) 
    disc = Discriminator().to(device) 
    gen.load_state_dict(torch.load(f'./metrics/{name}_gen_{num_steps}.pt', map_location=device))
    disc.load_state_dict(torch.load(f'./metrics/{name}_disc_{num_steps}.pt', map_location=device))
    gen.eval()
    disc.eval()

    if random:
        z = get_noise(len(images), 64, device=images.device)
    else:
        z = torch.zeros((len(images),64), device=images.device)
    im_features = disc.feature(images.view(len(images),1,28,28))
    for i in range(steps):
        z.requires_grad = True
        z.grad = None
        disc.zero_grad()
        gen.zero_grad()
        assert z.grad is None
        new_images = gen(z)
        res_loss = torch.abs(images - new_images).sum((1,2,3))
        disc_loss = torch.abs(im_features - disc.feature(new_images)).sum((1,2,3))
        loss = res_loss + disc_loss * lam
        loss_reduced = loss.sum()
        loss_reduced.backward(retain_graph=True)
        z.requires_grad = False
        #z_grad = z.grad.clone()
        z -= z.grad * step_size

        if i == steps - (steps // 5):
            step_size /= 10
        
    return loss.detach().cpu().numpy()




















### for conventional GAN without Wasserstein loss

class DiscriminatorGAN(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=16):
        super(DiscriminatorGAN, self).__init__()
        self.disc1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(im_chan, hidden_dim, 5, 2, padding=2)),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_dim, hidden_dim * 2, 5, 2, padding=1)),
        )
        self.disc2 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_dim * 2, 1, 5, 2)),
        )

    def feature(self, image):
        return self.disc1(image)

    def forward(self, image):
        disc_pred = self.disc2(self.disc1(image))
        return disc_pred.view(len(disc_pred), -1)
