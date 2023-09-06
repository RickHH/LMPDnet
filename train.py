import sys
import time

import cupy as cp


import pyparallelproj as ppp
from datasets1 import *
from model import *
import torch

savepath = ''
datapath = ''
dataset_name = ""

lr = 1e-6
start_epoch = 0
end_epoch = 500

devicenum = 0
device = torch.device("cuda:" + str(devicenum))

os.makedirs(savepath + "/%s/saved_models" % dataset_name, exist_ok=True)
savemodelpath = savepath + "/%s/saved_models" % dataset_name

f = open(savepath + "/%s/output.txt" % (dataset_name), 'a')

MSE = torch.nn.MSELoss()


model = LMNet(n_iter=8).to(device)
if start_epoch > 1:
    model.load_state_dict(torch.load(savemodelpath + "/net_params_{}.pkl".format(start_epoch-1)))

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
datasets = dataset(mode = 'train',datapath = datapath)
datasets_val = dataset(mode="validation",datapath = datapath)

with cp.cuda.Device(devicenum):
    scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16,1]),nmodules=np.array([28,1]))
    sino_params = ppp.PETSinogramParameters(scanner, ntofbins=17, tofbin_width=15.)
    img_dim = (128, 128, 1)
    voxsize = cp.array([2., 2., 2.])
    img_origin = (-(cp.array(img_dim) / 2) + 0.5) * voxsize
    proj = ppp.Projector(scanner, sino_params, img_dim, img_origin, voxsize, threadsperblock = 128, devicenum=devicenum)

    mempool = cp.get_default_memory_pool()
    train_losses = []
    train_psnrs = []
    train_ssims = []
    val_psnrs = []
    val_ssims = []

    valid_losses = []
    for epoch in range(start_epoch, end_epoch+1):
        model.train()
        train_loss = 0
        train_ssim_sum = 0
        train_psnr_sum = 0
        for i, batch in enumerate(datasets):
            events = batch["events"]
            image = batch["image"]

            sysG = proj.computesysG(events)
            sysG = torch.as_tensor(sysG, device=device)

            lst_events = torch.as_tensor(events[:,5], device=device, dtype=torch.float32).unsqueeze(1)

            image_tensor = torch.from_numpy(image).squeeze().to(device)
            output = model(sysG, lst_events).squeeze()
            MSE_loss = MSE(output, image_tensor) 

            loss = MSE_loss
            ssim_tmp = SSIM(output, image_tensor)
            psnr_tmp = psnr(output, image_tensor) 

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_ssim_sum += ssim_tmp.item()
            train_psnr_sum += psnr_tmp.item()

            del sysG
            mempool.free_all_blocks()
            torch.cuda.empty_cache()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d\%d] [loss %f]"
                % (
                    epoch,
                    end_epoch,
                    i,
                    len(datasets),
                    loss.item(),
                )
            )
        loss_mean = train_loss / (i + 1)
        train_ssim_mean = train_ssim_sum / (i + 1)
        train_psnr_mean = train_psnr_sum / (i + 1)
        print(
            "\r[Epoch %d/%d] [Batch %d\%d] [loss %f] [psnr %f] [ssim %f]"
            % (
                epoch,
                end_epoch,
                i,
                len(datasets),
                loss_mean,
                train_psnr_mean,
                train_ssim_mean,
            )
        )

        f.write(
            "\r[Epoch %d/%d] [Batch %d\%d] [loss %f] [psnr %f] [ssim %f]"
            % (
                epoch,
                end_epoch,
                i,
                len(datasets),
                loss_mean,
                train_psnr_mean,
                train_ssim_mean,
            )              
        )        
        train_losses.append(loss_mean)
        train_psnrs.append(train_psnr_mean)
        train_ssims.append(train_ssim_mean)
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0
            val_ssim_sum = 0
            val_psnr_sum = 0
            with torch.no_grad():
                for i,batch in enumerate(datasets_val):
                    events = batch["events"]
                    image = batch["image"]
                    sysG = proj.computesysG(events)
                    sysG = torch.as_tensor(sysG, device=device)

                    lst_events = torch.as_tensor(events[:,5], device=device, dtype=torch.float32).unsqueeze(1)

                    image_tensor = torch.from_numpy(image).squeeze().to(device)
                    output = model(sysG, lst_events).squeeze()
                    loss_val = MSE(output, image_tensor)
                    

                    ssim_tmp_val = SSIM(output, image_tensor)
                    psnr_tmp_val = psnr(output, image_tensor) 

                    val_loss += loss_val.item()
                    val_ssim_sum += ssim_tmp_val.item()
                    val_psnr_sum += psnr_tmp_val.item()

                    del sysG
                    mempool.free_all_blocks()
                
                val_loss_mean = val_loss / (i + 1)
                val_ssim_mean = val_ssim_sum / (i + 1)
                val_psnr_mean = val_psnr_sum / (i + 1)
                valid_losses.append(val_loss_mean)
                val_psnrs.append(val_psnr_mean)
                val_ssims.append(val_ssim_mean)
                print(
                    "\r[Epoch %d/%d] [val loss %f] [val psnr %f] [val ssim %f]\n"
                    %(
                        epoch,
                        end_epoch,
                        val_loss_mean,
                        val_psnr_mean,
                        val_ssim_mean,
                    )
                )

                f.write(
                    "\r[Epoch %d/%d] [val loss %f] [val psnr %f] [val ssim %f]\n"
                    % (
                        epoch,
                        end_epoch,
                        val_loss_mean,
                        val_psnr_mean,
                        val_ssim_mean,
                    )                
                )

            torch.save(model.state_dict(), savemodelpath + "/net_params_{}.pkl".format(epoch))
    
    torch.save(model.state_dict(), savemodelpath + "/net_params_final.pkl")
