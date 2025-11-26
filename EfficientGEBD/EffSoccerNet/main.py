import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER

import torch

from dataset import SoccerNet, SoccerNetClips, SoccerNetClipsTesting
from model import BaseModel
from train import trainer, test

from gebd_config import _C as cfg

import torch.distributed as dist
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR

import torch.backends.cudnn as cudnn
import random, shutil
from distribute import synchronize, all_gather, is_main_process
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler


def main(args, cfg, model_save_path):

    if is_main_process():
        logging.info("Parameters:")
        for arg in vars(args):
            logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split="train", framerate=args.framerate, chunk_size=args.chunk_size, receptive_field=args.receptive_field)
        # dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, split="valid", framerate=args.framerate, chunk_size=args.chunk_size, receptive_field=args.receptive_field)
    # dataset_Test  = SoccerNet(path=args.SoccerNet_path, features=args.features, split="test", framerate=args.framerate)
    dataset_Test  = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split="test", framerate=args.framerate, chunk_size=args.chunk_size, receptive_field=args.receptive_field)

    synchronize()

    # create model
    model = BaseModel(args, cfg).cuda()
    model.framerate = args.framerate
    # logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    if is_main_process():
        logging.info("Total number of parameters: " + str(total_params))

    start_epoch = -1
    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        start_epoch = state_dict['epoch']
        if is_main_process():
            print('Loaded from {}, Epoch: {}'.format(args.resume, start_epoch), flush=True)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        if not args.test_only:
            train_sampler = DistributedSampler(dataset_Train, shuffle=True)
            train_loader = torch.utils.data.DataLoader(dataset_Train,
                batch_size=args.batch_size, sampler=train_sampler,
                num_workers=args.max_num_worker, pin_memory=True)
        test_sampler = DistributedSampler(dataset_Test, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=args.batch_size, sampler=test_sampler,
            num_workers=args.max_num_worker, pin_memory=True)
    else:
        if not args.test_only:
            train_loader = torch.utils.data.DataLoader(dataset_Train,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.max_num_worker, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

    # training parameters
    if not args.test_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                     betas=(0.9, 0.999), eps=1e-08, 
                                     weight_decay=0, amsgrad=False)
        if args.scheduler == "ExponentialDecay":
            # scheduler = [args.LR, args.LR/1000]
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        elif args.scheduler == "ReduceLRonPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)
        
        if args.resume:
            for name, obj in [('optimizer', optimizer), ('scheduler', scheduler)]:
                if name in state_dict:
                    obj.load_state_dict(state_dict[name])
                    if is_main_process():
                        print('Loaded {} from {}'.format(name, args.resume), flush=True)

        # start training
        trainer(train_loader, test_loader, 
                model, optimizer, scheduler,
                train_game_name=dataset_Train.game_name,
                test_game_name=dataset_Test.game_name,
                max_epochs=args.max_epochs,
                dis_thres=args.dis_thres,
                model_save_path=model_save_path)

    best_model_path = os.path.join(model_save_path, "model.pth.tar")
    # print("loding?")
    if os.path.exists(best_model_path):
        print(f"loading {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])

    f1 = test(test_loader, model, dis_thres=args.dis_thres, game_name=dataset_Test.game_name)
    if is_main_process():
        logging.info("Best Performance at end of training " + str(f1))


def init_seeds(seed, cuda_deterministic=True):
    cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--expname", type=str, default='ckpt')
    parser.add_argument("--dis_thres", type=float, default=1)
    
    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="EffSoccerNet/data/R152_5fps",     help='Path for SoccerNet' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_5fps_TF2.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=100,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--test_only',   required=False, action='store_true', help='Perform testing only' )

    parser.add_argument('--framerate', required=False, type=int,   default=5,     help='Framerate of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=20,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--receptive_field', required=False, type=int,   default=2,     help='Temporal receptive field of the network (in seconds)' )
    parser.add_argument("--scheduler", required=False, type=str, default="ExponentialDecay", help="define scheduler")

    parser.add_argument('--batch_size', required=False, type=int,   default=16,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-3, help='Learning Rate' )
    parser.add_argument('--patience', required=False, type=int,   default=25,     help='Batch size' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=8, help='number of worker to load data')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    parser.add_argument('--config_file', type=str, default='EffSoccerNet/baseline.yaml')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=REMAINDER)

    args = parser.parse_args()

    if not args.local_rank:
        args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if torch.cuda.is_available():
        init_seeds(args.seed + args.local_rank)
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpus, rank=args.local_rank)
            dist.barrier()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    start_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    model_save_path = 'EffSoccerNet' + \
        f'{args.expname}_chunk{args.chunk_size}_k{args.receptive_field}_ep{args.max_epochs}_b{args.batch_size*args.num_gpus}_lr{args.LR}_disth{args.dis_thres}s'

    os.makedirs(model_save_path, exist_ok=True)
    log_path = os.path.join(model_save_path, f"log.log")

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    if is_main_process():
        logging.info('Starting main function')
    main(args, cfg, model_save_path)
    if is_main_process():
        logging.info(f'Total Execution Time is {time.time()-start} seconds')
