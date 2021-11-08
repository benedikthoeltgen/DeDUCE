
### run the JSMA algorithm (called by colab scripts)


import argparse
import torch

from DDU.resnet import resnet18
from CE.JSMA_utils import generate_JSMA_batch


parser = argparse.ArgumentParser(description='DDU training')

parser.add_argument('-s', '--manual_seed', type=int, default=0, metavar='MANUAL_SEED')
parser.add_argument('-md', '--model_dir', type=str, default='DDU/models/', metavar='MODEL_DIR')
parser.add_argument('-m', '--model_name', type=str, default=None, metavar='MODEL_NAME')
parser.add_argument('-d', '--device', type=str, default='cpu', metavar='DEVICE')
parser.add_argument('-dir', '--directory', type=str, default='', metavar='DIRECTORY')
parser.add_argument('--val', type=str, default=True, metavar='USE_VALSET')
parser.add_argument('--filename', type=str, default=None, metavar='FILENAME')

parser.add_argument('--CE_max_changes', type=int, default=5, metavar='CE_MAX_CHANGES')
parser.add_argument('--CE_max_iter', type=int, default=700, metavar='CE_MAX_ITER')
parser.add_argument('--CE_gamma', type=float, default=.5, metavar='CE_GAMMA')    #target confidence
parser.add_argument('--CE_delta', type=float, default=.2, metavar='CE_DELTA')     #step size
parser.add_argument('--CE_pixels', type=int, default=1, metavar='CE_PIXELS')  #number of pixels changed at a time

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

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

if args.cuda:
    print("Model on GPU")
    model.cuda()
    args.device = 'cuda'


### load data
name = '_vbatch' if args.val else ''
filename = args.filename
X = torch.load(args.directory + f'{filename}_X.pt')
labels = torch.load(args.directory + f'{filename}_y.pt')





#################################################################################################
#####     generate CEs systematically
#################################################################################################



    
p = f'p{args.CE_pixels}'


    
iter_matrix = torch.empty((X.shape[0], 10)).to(args.device)
CE_matrix = torch.empty((X.shape[0], 10, 28*28)).to(args.device)
    
for target_class in range(10):
    
    print(f'target class: {target_class}')
    
    X_new, iterations = generate_JSMA_batch(X, labels, target_class, model, args.device, 
                                            args.CE_gamma, args.CE_delta, args.CE_max_iter, args.CE_pixels)
    
    CE_matrix[:,target_class] = X_new
    iter_matrix[:,target_class] = iterations

torch.save(iter_matrix.to('cpu'), args.directory + f'JSMA_iter{name}_c0{p}.pt')
torch.save(CE_matrix.detach().to('cpu'), args.directory + f'JSMA_arr{name}_c0{p}.pt')
        
        
        
        