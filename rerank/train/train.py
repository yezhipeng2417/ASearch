import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME
from config import Args
from classi_model import modeling,focal_loss
from classi_model.trainer import Trainer
from data_modules.dataloder import data_loaders
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def train(rank, n_gpus_per_node):
    args = Args()
    device = f'cuda:{rank}'
    # device = 'cuda'
    dist.init_process_group(
        backend='nccl', init_method='tcp://localhost:2333', world_size=n_gpus_per_node, rank=rank)
    torch.cuda.set_device(rank)

    train_dataloder = data_loaders(args.train_path, args.tokenizer,ddp = True,bath = args.batch,num_worker = args.num_workers)
    test_dataloder = data_loaders(args.test_path, args.tokenizer,ddp = True,bath = args.batch,num_worker = args.num_workers)

    model = modeling.get_model(args).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = focal_loss.FocalLoss(device,gamma=2)

    # train

    trainer = Trainer(args=args,
                      model=model,
                      train_data=train_dataloder,
                      dev_data=test_dataloder,
                      loss_func=loss_func,
                      optimizer=optimizer,
                      rank=rank,
                      device=device)
    trainer.train()


if __name__ == '__main__':
    mp.spawn(train, nprocs=2, args=(2,))
