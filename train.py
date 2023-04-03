import os
import random
import time
from datetime import datetime

import numpy as np
import argparse
from yacs.config import CfgNode

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from tools.pseudo_face_data import faces_data
from tools.utils import save_tensor_image, AverageMeter
from models.face_model import Face_Model

main_parse = argparse.ArgumentParser()
main_parse.add_argument("yaml", type=str)
main_parse.add_argument("--port", type=int, default=2357, required=False)
main_args = main_parse.parse_args()
with open(main_args.yaml, "rb") as cf:
    CFG = CfgNode.load_cfg(cf)
    CFG.freeze()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_args.port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_train_loader(trainset, world_size, batch_size):
    if world_size <= 1:
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=CFG.DATA.NUM_WORKERS, pin_memory=True)
    elif world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler, num_workers=CFG.DATA.NUM_WORKERS, pin_memory=True)

def main(rank, world_size, cpu=False):
    if cpu:
        rank = torch.device("cpu")
    elif world_size > 1:
        setup(rank, world_size)
    last_device = world_size - 1
    batch_per_gpu = CFG.SR.BATCH_PER_GPU

    model = Face_Model(rank, CFG, world_size > 1)
    trainset = faces_data(data_lr=os.path.join(CFG.DATA.FOLDER, "LOW/LR"), data_hr=os.path.join(CFG.DATA.FOLDER, "HIGH/HR"),   img_range=CFG.DATA.IMG_RANGE, rgb=CFG.DATA.RGB)
    #trainset = faces_data(data_lr=os.path.join(CFG.DATA.FOLDER, "LOW/data_LR"), data_hr=os.path.join(CFG.DATA.FOLDER, "HIGH/data_HR"),   img_range=CFG.DATA.IMG_RANGE, rgb=CFG.DATA.RGB)
    loader = get_train_loader(trainset, world_size, batch_per_gpu)
    testset = faces_data(data_lr=os.path.join(CFG.DATA.FOLDER, "testset"), data_hr=None, b_train=False, shuffle=False, img_range=CFG.DATA.IMG_RANGE, rgb=CFG.DATA.RGB)

    #Calculates the number of iterations required to complete the training process based on the maximum number of iterations (MAX_ITER) specified in the configuration file (CFG.OPT) and the number of batches in the data loader (len(loader)).
    end_ep = int(np.ceil(CFG.OPT.MAX_ITER / len(loader))) + 1   # len(loader): This returns the number of batches in the data loader. MAX_ITER: 100000
    print(f"Lujunn number_epoch is: {end_ep}")
    test_freq = max([end_ep // 10, 1])

    if rank == last_device:
        net_save_folder = os.path.join(CFG.EXP.OUT_DIR, "nets")
        img_save_folder = os.path.join(CFG.EXP.OUT_DIR, "imgs")
        os.makedirs(net_save_folder, exist_ok=True)
        os.makedirs(img_save_folder, exist_ok=True)
        print("Output dir: ", CFG.EXP.OUT_DIR)
        print(f"Batch_size: {batch_per_gpu * world_size}, Batch_size per GPU: {batch_per_gpu}")
        print(f"Max epoch: {end_ep - 1}, Total iteration: {(end_ep - 1) * len(loader)}, Iterations per epoch: {len(loader)}, Test & Save epoch: every {test_freq} epoches")

    loss_avgs = dict()
    
    ##################################
    model_start_time = time.time()  # measure the start time of model processing
    ##################################
    
    for ep in range(1, end_ep):
        ##################################
        epoch_start_time = time.time()  # measure the start time of epoch processing
        ##################################
        model.mode_selector("train")
        for b, batch in enumerate(loader):
            batch_start_time = time.time()  # measure the start time of batch processing
            lrs = batch["lr"].to(rank)
            hrs = batch["hr"].to(rank)
            zs = batch["z"].to(rank)
            hr_downs = batch["hr_down"].to(rank)
            losses = model.train_step(hrs, lrs, hr_downs, zs)
            info = f"  {model.n_iter}({ep}/{end_ep-1}):"
            for i, itm in enumerate(losses.items()):
                if itm[0] not in loss_avgs.keys():
                    loss_avgs[itm[0]] = AverageMeter(itm[1])
                else:
                    loss_avgs[itm[0]].update(itm[1])
                info += f", {itm[0]}={loss_avgs[itm[0]].get_avg():.3f}" if i > 0 else f" {itm[0]}={loss_avgs[itm[0]].get_avg():.3f}"
            print(info + "\r", end="")
            model.lr_decay_step(True)
            ##############################################################################
            ##############Calculate the bacth training time and bacththroughput###########
            batch_end_time = time.time()  # measure the end time of batch processing
            batch_training_time = batch_end_time - batch_start_time  # calculate the batch training time
            batch_size_amount = batch_per_gpu * world_size
            batch_throughput = batch_size_amount / batch_training_time
            print(f"Lujunnnn {b}th batch training time: {batch_training_time:.2f}s")
            print(f"Lujun {b}th batch_throughput: {batch_throughput:.2f}s")
            
            
            if b % 1 == 0:
                batch_data_dict = {'batch_start_time': batch_start_time, 'batch_training_time': batch_training_time, 'batch_end_time': batch_end_time, 'batch_throughput': batch_throughput}
                torch.save(batch_data_dict, 'batch_training_time_throughput.pt')            
            #############################################################################
            #############################################################################
            
            
        ###########################################################################################   
        ##############Calculate the epoc training time and epoc throughput#########################
        epoch_end_time = time.time()  # measure the end time of epoch processing
        epoch_training_time = epoch_end_time - epoch_start_time   # calculate the epoch training time
        examples_per_sec = len(trainset) / epoch_training_time  # calculate the epoch throughput (examples processed per second); len(trainset) is the number of train samples
        epoch_throughput = examples_per_sec
        
        print(f"{ep}th epoch training time: {epoch_training_time:.2f}s")
        print(f"throughput Lujunnnn {examples_per_sec:.2f}")
        print(f"\nEpoch {ep}/{end_ep-1} took {epoch_training_time:.2f} seconds, epoch throughput={examples_per_sec:.2f} ex/s") 
        epoch_training_troughput_data_dict = {'epoch_training_time': epoch_training_time,'epoch_throughput': epoch_throughput}
        torch.save(epoch_training_troughput_data_dict, 'epoch_training_time_throughput.pt',epoch_throughput_data_dict, 'epoch_throughput.pt') 
        ##########################################################################################
        ##########################################################################################
    
        if ep % 1 == 0 and rank == last_device:
            print(f"\nTesting and saving: {datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}")
            model.net_save(net_save_folder)
            model.mode_selector("eval")
            for b in range(len(testset)):
                if b > 100: break
                lr = testset[b]["lr"].unsqueeze(0).to(rank)
                y, sr, _ = model.test_sample(lr)
                save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_y.png"), y, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
                save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_sr.png"), sr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
                save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_lr.png"), lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
        if world_size > 1: dist.barrier()
 
    ############################################################################
    model_end_time = time.time()
    model_training_time = model_end_time - model_start_time
    print(f"Total training time: {model_training_time:.2f}s")
    data_dict = {'model_start_time': model_start_time, 'model_training_time': model_training_time, 'model_end_time': model_end_time, 'total_epoch':end_ep}
    torch.save(data_dict, 'log_model_training_time.pt') 
    #############################################################################
            
            
    if rank == last_device:
        print("\nFinal test and save")
        model.net_save(net_save_folder, True)
        model.mode_selector("eval")
        for b in range(len(testset)):
            lr = testset[b]["lr"].unsqueeze(0).to(rank)
            y, sr, _ = model.test_sample(lr)
            save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_y.png"), y, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
            save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_sr.png"), sr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
            save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_lr.png"), lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
    if world_size > 1: dist.barrier()
    if world_size > 1:cleanup()

if __name__ == "__main__":
    random_seed = CFG.EXP.SEED
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print("torch.cuda.device_count()",torch.cuda.device_count())

        ###############################################
        print("torch.cuda.is_available",torch.cuda.is_available())
        print("Lujun11")
        ######################################
    if n_gpus <= 1:
        print("single proc.", f", time: {datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}")
        main(0, n_gpus, cpu=(n_gpus == 0))
    elif n_gpus > 1:
        print(f"multi-gpu: {n_gpus}", f", time: {datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}")
        mp.spawn(main, nprocs=n_gpus, args=(n_gpus, False), join=True)
    print("fin.")
