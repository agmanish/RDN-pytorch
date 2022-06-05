import argparse
import json

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
from models import RDN
from utils import convert_rgb_to_y, denormalize, calc_psnr,  calc_ssim, AverageMeter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--test-img-dir', type=str, required=True)
    parser.add_argument('--op-dir', type=str, required=True)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    
    ip_path=args.test_img_dir
    filenames=os.listdir(ip_path)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    psnr_dict={}
    ssim_dict={}

    for file in filenames:
          image = pil_image.open(os.path.join(ip_path,file)).convert('RGB')

          orig= np.array(image).astype(np.float32)
          orig=convert_rgb_to_ycbcr(orig)  
          orig = orig[..., 0]
          orig /= 255.
          orig = torch.from_numpy(orig).to(device)
          oig = orig.unsqueeze(0).unsqueeze(0)


          image_width = (image.width // args.scale) * args.scale
          image_height = (image.height // args.scale) * args.scale
          image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
          image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
          image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
          op_name=os.path.join(args.op_dir,file)
          image.save(op_name.replace('.', '_bicubic_x{}.'.format(args.scale)))

          image = np.array(image).astype(np.float32)
          ycbcr = convert_rgb_to_ycbcr(image)

          y = ycbcr[..., 0]
          y /= 255.
          y = torch.from_numpy(y).to(device)
          y = y.unsqueeze(0).unsqueeze(0)


          with torch.no_grad():
              preds = model(y).clamp(0.0, 1.0)
          #print(orig.size(),preds.size())
          orig=orig.view(1,1,orig.size()[0],orig.size()[1])
          #print(orig.size(),preds.size())
          psnr = calc_psnr(orig, preds)
          ssim = calc_ssim(orig, preds)
          
          print('PSNR: {:.2f}'.format(psnr))
          print('SSIM: {:.2f}'.format(ssim))

          preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

          output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
          output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
          output = pil_image.fromarray(output)
          output.save(op_name.replace('.', '_srcnn_x{}.'.format(args.scale)))
          psnr_dict[file] = psnr.item()
          ssim_dict[file] = ssim.item()
          #print(psnr_dict[file], ssim_dict[file])
    test_metrics={
        "psnr_vs_epoch":psnr_dict,
        "ssim_vs_epoch":ssim_dict,
    }

    json_path=args.op_dir+"/test_metrics.json"
    with open(json_path, "w") as outfile:
        json.dump(test_metrics, outfile)
