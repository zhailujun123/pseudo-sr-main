import os
import argparse
from yacs.config import CfgNode

import torch
from models.face_model import Face_Model
from tools.pseudo_face_data import faces_data
from tools.utils import save_tensor_image, AverageMeter


model_path = "./results/faces/nets/nets_106512.pth"  
test_data_path = "./Dataset/testset_Historical"          
img_save_folder = "./results/faces/imgs/"
device = 0

main_parse = argparse.ArgumentParser()
main_parse.add_argument("yaml", type=str)
main_args = main_parse.parse_args()
with open(main_args.yaml, "rb") as cf:
    CFG = CfgNode.load_cfg(cf)
    CFG.freeze()

testset = faces_data(data_lr=test_data_path, data_hr=None, b_train=False, shuffle=False, img_range=CFG.DATA.IMG_RANGE, rgb=CFG.DATA.RGB)

model = Face_Model(device, CFG)
model.net_load(model_path)
model.mode_selector("eval")

print(len(testset))

for b in range(len(testset)):
    if b > 20: break
    lr = testset[b]["lr"].unsqueeze(0).to(device)
    y, sr, _ = model.test_sample(lr)
    save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_y.png"), y, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
    save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_sr.png"), sr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
    save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_lr.png"), lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
