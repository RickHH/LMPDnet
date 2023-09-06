import cupy as cp
import torch

import pyparallelproj as ppp
from datasets1 import *
from model import *



savepath = ''
datapath = ''
dataset_name = ""

end_epoch = 300

devicenum = 0
device = torch.device("cuda:" + str(devicenum))

save_result_path = savepath + "/%s" % dataset_name
savemodelpath = savepath + "/%s/saved_models" % dataset_name


model = LMNet(n_iter=8).to(device)
model.load_state_dict(torch.load(savemodelpath + "/net_params_{}.pkl".format(end_epoch), map_location=device))
model.eval()

datasets = dataset(mode="test",datapath=datapath)

result = np.zeros((128,128,5,2))

with cp.cuda.Device(devicenum):
    scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16,1]),
                                        nmodules=np.array([28,1]))
    sino_params = ppp.PETSinogramParameters(scanner, ntofbins=17, tofbin_width=15.)
    img_dim = (128, 128, 1)
    voxsize = cp.array([2., 2., 2.])
    img_origin = (-(cp.array(img_dim) / 2) + 0.5) * voxsize
    proj = ppp.Projector(scanner, sino_params, img_dim, img_origin, voxsize, devicenum=devicenum)

    for i, batch in enumerate(datasets):
        
        events = batch["events"]
        image = batch["image"]
        sysG = proj.computesysG(events)
        sysG = torch.as_tensor(sysG, device=device)
        lst_events = torch.as_tensor(events[:,5], device=device, dtype=torch.float32).unsqueeze(1)
        image_tensor = torch.from_numpy(image).squeeze().to(device)
        output = model(sysG, lst_events).squeeze().cpu().detach().numpy()
        result[:,:,i,0] = output
        result[:,:,i,1] = image

        del sysG
        cp.get_default_memory_pool().free_all_blocks()

    
    sio.savemat(save_result_path + '/{}_{}epo.mat'.format(dataset_name,end_epoch), {'result_{}epo'.format(end_epoch):result})

        
