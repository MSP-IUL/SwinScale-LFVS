import time
import argparse
import torch.backends.cudnn as cudnn
from utils.utils import *
from model import TRNS
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import scipy.io as sio
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=2, help="angular resolution")
    parser.add_argument("--angout", type=int, default=7, help="angular resolution")
    parser.add_argument("--factor", type=int, default=7, help="upscale factor")
    parser.add_argument('--testset_dir', type=str, default='/home/mzrdu/Downloads/Learning purpose/Diffusion model/HCI/Test_HCI_2x2-7x7/')
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=32, help="The stride between two test patches is set to patchsize/2")
    parser.add_argument('--model_path', type=str, default='checkpoint/HCI_TRNS.pth.tar')
    parser.add_argument('--save_path', type=str, default='Results/')
    return parser.parse_args()

def load_checkpoint(model, model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded successfully from '{model_path}'")
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        raise


def test(cfg, test_Names, test_loaders):
    net = TRNS(cfg)
    net.to(cfg.device)
    
    cudnn.benchmark = True

    ##### get input index ######
    ind_all = np.arange(cfg.angout * cfg.angout).reshape(cfg.angout, cfg.angout)
    delt = (cfg.angout - 1) // (cfg.angin - 1)
    ind_source = ind_all[0:cfg.angout:delt, 0:cfg.angout:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))

    model_path = os.path.abspath(cfg.model_path)
    print("Model file path:", model_path)

    if os.path.isfile(model_path):
        print("Loading model from '{}'".format(model_path))
        load_checkpoint(net, model_path, cfg.device)
    else:
        print(f"=> No model found at '{model_path}'")
        directory = os.path.dirname(model_path)
        print(f"Contents of the directory '{directory}':")
        print(os.listdir(directory))
        return

    results = []

    with torch.no_grad():
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test, scene_results = inference(test_loader, test_name, net, ind_source, cfg)
            results.extend(scene_results)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass

    # Save results to Excel
    df = pd.DataFrame(results, columns=['Dataset', 'Scene', 'PSNR', 'SSIM'])
    df.to_excel(cfg.save_path + 'scene_metrics.xlsx', index=False)
    print(f'Results saved to {cfg.save_path}scene_metrics.xlsx')

def inference(test_loader, test_name, net, ind_source, cfg):
    psnr_iter_test = []
    ssim_iter_test = []
    scene_results = []

    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        label = label.squeeze()
        #print(data.shape)

        uh, vw = data.shape
        h0, w0 = uh // cfg.angin, vw // cfg.angin
        subLFin = LFdivide(data, cfg.angin, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angout * cfg.patchsize, cfg.angout * cfg.patchsize)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, cfg.angout, cfg.patchsize, cfg.stride, h0, w0)
        psnr, ssim = cal_metrics(label, outLF, cfg.angout, ind_source)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        scene_name = test_loader.dataset.file_list[idx_iter][:-3]
        scene_results.append((test_name, scene_name, psnr, ssim))

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists):
            os.makedirs(cfg.save_path + test_name)

        sio.savemat(cfg.save_path + test_name + '/' + scene_name + '.mat',
                    {'LF': outLF.numpy()})
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test, scene_results

def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
