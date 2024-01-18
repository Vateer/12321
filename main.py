import argparse, wandb
import torch
from src.utils import *
from src.client import *
from src.server import *
from datetime import datetime

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="DualFL", help="algorithm")
    parser.add_argument('--rounds', type=int, default=160, help="rounds of training")
    parser.add_argument('--ft_rounds', type=int, default=40, help="rounds of fine-tune")
    parser.add_argument('--clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--distribution', type=str, default="pat", help="data distribution(iid, pat, dir)")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--ft_frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--savedir', type=str, default='./save/', help='save directory')
    parser.add_argument('--logdir', type=str, default='./log/', help='log directory')
    parser.add_argument('--gpu', type=str, default="3", help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--load_initial', type=str, default='', help='define initial model path')
    parser.add_argument('--test_frac', type=float, default=0.0, help='test data rate')
    parser.add_argument('--dir_alpha', type=float, default=0.1, help='alpha in dirichlet distribution')
    parser.add_argument('--svg_k', type=int, default=3, help='number of chose svg vector per class')
    parser.add_argument('--hc_clustering', type=float, default=0.007, help='hierarchical clustering')
    parser.add_argument('--hc_type', type=str, default="average", help='hierarchical clustering linkage type ( maximum, minimum, average )')
    parser.add_argument('--optics_xi', type=float, default=0.01, help='xi of OPTICS algorithm')
    parser.add_argument('--optics_min_sample', type=int, default=9, help='min_sample of OPTICS algorithm')
    parser.add_argument('--log', type=bool, default=False, help='need to log?')
    parser.add_argument('--wandb', type=bool, default=False, help='need to upload result to wandb?')
    parser.add_argument('--wandb_project', type=str, default="DualFL", help='name of your wandb project')
    parser.add_argument('--wandb_name', type=str, default="exp", help='name of your wandb exp')
    parser.add_argument('--recluster_delta', type=int, default=50, help='check if recluster is needed')
    parser.add_argument('--lc_threshold', type=float, default=2, help='lc_threshold')
    parser.add_argument('--std_threshold', type=float, default=10, help='lc_threshold')
    parser.add_argument('--just_global', type=bool, default=False, help='just launch global training process')
    parser.add_argument('--aux_rate', type=float, default=0.01, help='Auxiliary dataset ratio (0.1 means 10%)')
    parser.add_argument('--shannon_threshold', type=float, default=0.5, help='shannon threshold')
    parser.add_argument('--tag', type=str, default=" ", help='Log tag')
    parser.add_argument('--lr_decay', type=float, default=0, help='Learning rate decay')
    parser.add_argument('--pat_cls', type=int, default=2, help='class number in pat distribution')
    parser.add_argument('--model', type=str, default='LeNet', help='model name')
    parser.add_argument('--dataset', type=str, default='fmnist', 
                        help="name of dataset: fmnist, cifar10, cifar100, svhn, mix4")

    args = parser.parse_args()
    args.lr_now = args.lr
    return args

#initialization
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
setup_seed(args.seed)

_ = input("log {}, wandb {} , input any key to begin \n".format(args.log, args.wandb))

#wandb setup
if args.wandb == True:
    now = datetime.now()
    timestamp = now.strftime("%m%d%H%M")
    args.wandb_name = args.alg+"_"+args.model+"_"+str(args.rounds)+"_"+str(args.seed)+"_"+timestamp+"_"+"rc0"
    if args.just_global == True:
        args.wandb_name = args.alg+"_"+args.model+"_"+str(args.rounds)+"_"+str(args.clients)+"_"+str(args.frac)+"_"+str(args.seed)+"_"+"FT"
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

split_from_trainset = False
if args.test_frac > 0:
    split_from_trainset = True

#data distribution
datas, labels, test_datas, test_labels = handle_data(args=args, join = split_from_trainset)
if args.alg == "DualFL":
    auxiliary_datas, auxiliary_labels, datas, labels = create_auxiliary_dataset(args, datas, labels)
data_splits = partition_data(args=args, datas=datas, labels=labels)
if split_from_trainset == False:
    test_splits = partition_data(args=args, datas=test_datas, labels=test_labels)

#client initialization
clients = []
for i in range(args.clients):
    client = eval("{}Client".format(args.alg))
    if args.dataset == "mix4":
        nums = [32, 15, 28, 25]
        j = i
        k = 0
        for num in nums:
            if num > j:
                break
            j -= num
            k += 1
        train_loader, test_loader = make_dataloader(args=args, data_split=data_splits[i], test_datas=test_datas[k], test_labels=test_labels[k], split_from_trainset=split_from_trainset)
    else:
        train_loader, test_loader = make_dataloader(args=args, data_split=data_splits[i], test_datas=test_datas, test_labels=test_labels, split_from_trainset=split_from_trainset)
    # print(test_loader.dataset.__len__())
    clients.append(client(args, train_loader, test_loader))

#server initialization1
server = eval("{}Server".format(args.alg))(args, clients)
if args.alg == "DualFL":
    server.load_auxiliary_dataset(auxiliary_datas, auxiliary_labels)
server.run()
# server.run_global_model_training(r"")
# server.inference(r"")
# server.inference_global(r"", get_test_loader(args, test_datas, test_labels))
#wandb end
if args.wandb == True:
    wandb.finish()
