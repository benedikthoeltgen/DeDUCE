
### run DeDUCE to generate counterfactuals


import argparse
import torch

from DDU.train_utils import get_data
from DDU.resnet_DDU import resnet18 as resnetDDU
from CE.gmm_utils import get_embeddings, gmm_fit_ex
from CE.DeDUCE_utils import generate_CE_batch


parser = argparse.ArgumentParser(description='DDU training')

parser.add_argument('-s', '--manual_seed', type=int, default=0, metavar='MANUAL_SEED')
parser.add_argument('-md', '--model_dir', type=str, default='DDU/models/', metavar='MODEL_DIR')
parser.add_argument('-m', '--model_name', type=str, default=None, metavar='MODEL_NAME')
parser.add_argument('-le', '--load_gmm', type=bool, default=False, metavar='LOAD_GMM')
parser.add_argument('-dev', '--device', type=str, default='cpu', metavar='DEVICE')
parser.add_argument('-dir', '--directory', type=str, default='', metavar='DIRECTORY')
parser.add_argument('--filename', type=str, default=None, metavar='FILENAME')

parser.add_argument('-c', '--coeff', type=float, default=4., metavar='COEFF')
parser.add_argument('--max_changes', type=int, default=5, metavar='MAX_CHANGES')
parser.add_argument('--max_iter', type=int, default=700, metavar='MAX_ITER')
parser.add_argument('--gamma', type=float, default=.5, metavar='GAMMA')    #target confidence
parser.add_argument('--delta', type=float, default=.2, metavar='DELTA')     #step size
parser.add_argument('--momentum', type=float, default=.6, metavar='MOMENTUM')  #momentum
parser.add_argument('--pixels', type=int, default=1, metavar='PIXELS')  #number of pixels changed at a time
parser.add_argument('--loss_coeff', type=int, default=1, metavar='LOSS_COEFF')
parser.add_argument('--lossgrad', type=bool, default=False, metavar='LOSS_GRAD') #use grad of plain loss

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

if args.manual_seed is not None:
    torch.manual_seed(args.manual_seed)




#################################################################################################
#####     setup model and fit GMM
#################################################################################################



### load model
coeff = args.coeff
momentum= args.momentum

if args.model_name is None:
    args.model_name = 'model_c' + str(int(coeff))

model = resnetDDU(coeff = coeff)

    
model.load_state_dict(torch.load(args.model_dir + args.model_name + '.pt', map_location=torch.device('cpu')))
model.eval()

if args.cuda:
    args.device = 'cuda'
    model.cuda()
    print("Model on GPU")


### load data

trainloader, _, _ = get_data('data/', 100, seed=1)


### fit or load GMM

num_dim = 512

if args.load_gmm:
    means, cov_matrix = torch.load(args.model_dir + args.model_name + '_gmm.pt', map_location=(args.device))
    gmm = torch.distributions.MultivariateNormal(loc=means.to(args.device),
                                                 covariance_matrix=cov_matrix.to(args.device))
else:
    embeddings, labels = get_embeddings(model, trainloader, num_dim=num_dim, dtype=float, 
                                        device=args.device, storage_device=args.device)
    gmm, jitter_eps = gmm_fit_ex(embeddings=embeddings, labels=labels, num_classes=10, gmm_type="gda")
    print('jitter_eps: ', jitter_eps)
    torch.save((gmm.mean, gmm.covariance_matrix), args.model_dir + args.model_name + '_gmm.pt')
    







#################################################################################################
#####     generate CEs systematically
#################################################################################################


### load data
X = torch.load(args.directory + f'{args.filename}_X.pt')
labels = torch.load(args.directory + f'{args.filename}_y.pt')

    
iter_matrix = torch.empty((X.shape[0], 10)).to(args.device)
CE_matrix = torch.empty((X.shape[0], 10, 28*28)).to(args.device)
    
for target_class in range(10):
    
    print(f'target class: {target_class}')
    
    X_new, iterations, _ = generate_CE_batch(X, labels, target_class, model, gmm, args.gamma, args.delta,
                                             args.max_changes, args.max_iter, momentum,
                                             args.loss_coeff, args.pixels, args.lossgrad)
    
    CE_matrix[:,target_class] = X_new
    iter_matrix[:,target_class] = iterations


### save
cl_cl = 'CL' if args.lossgrad else 'cl'
cl = cl_cl + f'{args.loss_coeff}'
m = f'm{int(momentum*10)}'
p = f'p{args.pixels}'
c = f'c{int(coeff)}'

torch.save(iter_matrix.to('cpu'), args.directory + f'DeDUCE_iter_{c}{m}{cl}{p}.pt')
torch.save(CE_matrix.detach().to('cpu'), args.directory + f'DeDUCE_arr_{c}{m}{cl}{p}.pt')
        
