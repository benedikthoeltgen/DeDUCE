
### train the resnets

import argparse
import torch

from DDU.train_utils import get_data, train
from DDU.resnet import resnet18
from DDU.resnet_DDU import resnet18 as resnetDDU



parser = argparse.ArgumentParser(description='DDU training')

parser.add_argument('-s', '--manual_seed', type=int, metavar='MANUAL_SEED')
parser.add_argument('-bs', '--batch_size', type=int, default=128, metavar='BATCH_SIZE')
parser.add_argument('-od', '--out_dir', type=str, default='DDU/models/', metavar='OUT_DIR')
parser.add_argument('-m', '--model_name', type=str, metavar='OUT_DIR')
parser.add_argument('-e', '--epochs', type=int, default=50, metavar='EPOCHS')
parser.add_argument('-c', '--sn_coeff', type=float, default=3., metavar='SN_COEFF')
parser.add_argument('-dd', '--data_dir', type=str, default=None, metavar='DATA_DIR')

args = parser.parse_args()

args.cuda = torch.cuda.is_available()



if args.manual_seed is not None:
    torch.manual_seed(args.manual_seed)



### if no data directory specified, take /scratch-ssd/oatml/data if on oatml cluster and ./data otherwise

if args.data_dir is None:
    args.data_dir = './data'



### get data

trainloader, valloader, _ = get_data(args.data_dir, args.batch_size, args.manual_seed)


### initialise model

if args.sn_coeff == 0.:
    model = resnet18()
    print('Basic ResNet without spectral normalisation')
else:
    model = resnetDDU(coeff = args.sn_coeff)
    print('ResNet with spectral normalisation coeff', args.sn_coeff)
    

if args.cuda:
    print("Model on GPU")
    model.cuda()


### train model
model = train(model, args, trainloader, valloader)


### save model
torch.save(model.state_dict(), args.out_dir + args.model_name + '.pt')
