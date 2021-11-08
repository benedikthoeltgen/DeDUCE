
### run REVISE algorithm to generate counterfactuals from Joshi et al. 2019


import argparse
import torch

from DDU.resnet import resnet18
from CE.REVISE_utils import generate_CE_batch

from CE.VAE import VAE


parser = argparse.ArgumentParser(description='REVISE ALGORITHM')

parser.add_argument('-s', '--manual_seed', type=int, default=0, metavar='MANUAL_SEED')
parser.add_argument('-md', '--model_dir', type=str, default='DDU/models/', metavar='MODEL_DIR')
parser.add_argument('-m', '--model_name', type=str, default=None, metavar='MODEL_NAME')
parser.add_argument('-dir', '--directory', type=str, default='', metavar='DIRECTORY')
parser.add_argument('--val', type=bool, default=True, metavar='USE_VALSET')
parser.add_argument('--filename', type=str, default=None, metavar='FILENAME')

parser.add_argument('--max_iter', type=int, default=50000, metavar='CE_MAX_ITER')
parser.add_argument('--gamma', type=float, default=.5, metavar='GAMMA')    #target confidence
parser.add_argument('--step_size', type=float, default=1e-5, metavar='STEP_SIZE')     #step size
parser.add_argument('--lam', type=float, default=1, metavar='LAMBDA')  #loss fct hyperparam

args = parser.parse_args()

cuda = torch.cuda.is_available()

if args.manual_seed is not None:
    torch.manual_seed(args.manual_seed)




#################################################################################################
#####     setup model and fit GMM
#################################################################################################



### load model
if args.model_name is None:
    args.model_name = 'model_c0'

model = resnet18()

    
model.load_state_dict(torch.load(args.model_dir + args.model_name + '.pt', map_location=torch.device('cpu')))
model.eval()

### load VAE

vae = VAE()
vae.load_state_dict(torch.load('CE/VAE_model.pt', map_location=torch.device('cpu')))
vae.eval()


device = 'cpu'
if cuda:
    print("Model on GPU")
    model.cuda()
    vae.cuda()
    device = 'cuda'


### load data

name = '_vbatch' if args.val else ''
filename = args.filename
X = torch.load(args.directory + f'{filename}_X.pt')
labels = torch.load(args.directory + f'{filename}_y.pt')


    







#################################################################################################
#####     generate CEs systematically
#################################################################################################


    
iter_matrix = torch.empty((X.shape[0], 10)).to(device)
CE_matrix = torch.empty((X.shape[0], 10, 28*28)).to(device)
    
for target_class in range(10):
    
    print(f'target class: {target_class}')
    
    X_new, iterations = generate_CE_batch(X.to(device), target_class, model, vae, 
                                             args.lam, args.step_size, args.max_iter, args.gamma)
    
    CE_matrix[:,target_class] = X_new
    iter_matrix[:,target_class] = iterations

import math
lam = int(math.log10(args.lam))
step_size = int(math.log10(args.step_size))

torch.save(iter_matrix.to('cpu'), args.directory + f'REVISE_l{lam}s{step_size}_iter{name}_c0.pt')
torch.save(CE_matrix.detach().to('cpu'), args.directory + f'REVISE_l{lam}s{step_size}_arr{name}_c0.pt')
        
