from mllogger import *
from lbae import *

def get_params():
    hps = Params()
    hps.zsize = 100
    hps.batch_size = 64
    hps.batch_size_test = 64
    hps.epochs_max = 3900 
    hps.lr[0] = 1e-3
    hps.img_size = 28
    # hps.img_crop_size = None
    hps.channels = 1
    hps.model_conv = False
    hps.shared_weights = False
    hps.workers = 10
    hps.print_every_batch = 100
    hps.keep_last_models = 10
    hps.binary_reco_loss = True
    hps.corrupt_method_test = 'blank'
    hps.corrupt_args_test = []
    hps.corrupt_method = 'blank'
    hps.corrupt_args = []
    hps.vae_model = None
    hps.mem_model = None

    hps.parallel = False
    hps.vae = False
    hps.sample_method = 'kde'
    hps.kde_bw = 0.1
    hps.gen_knn = False
    hps.znoise_std = 0
    hps.zdrop = 0
    hps.l2 = 0
    hps.gen_imgs = 64 # Number of generated images
    hps.zsize_cont = 0 
    hps.zsize_cont_enc = 0
    hps.set_attr = -1

    hps.cfg = 'celeba_bae'
    # hps.cfg = 'celeba_vae'

    if hps.cfg == 'celeba_vae':
        hps.dataset = 'celeba'
        hps.exp_suffix = 'm173'
        hps.vae = True
        hps.lr[0] = 1e-3
        hps.parallel =True 
        hps.batch_size = 512
        hps.batch_size_test = 1024
        hps.sample_method = 'random'
        hps.gen_imgs = 5000
        hps.img_size = 64 
        hps.kl_weight = 64
        hps.zsize = 64
        hps.channels = 3
        hps.binary_reco_loss = False
        hps.vae_model = 'ConvResBlock32'

    if hps.cfg == 'celeba_bae':
        hps.dataset = 'celeba'
        hps.exp_suffix = 'm170-3'
        hps.vae = False
        hps.lr[0] = 1e-3
        hps.parallel =True 
        hps.batch_size = 512
        hps.batch_size_test = 1024
        hps.sample_method = 'cov'
        hps.gen_imgs = 5000
        hps.interpolate_steps = 10
        hps.img_size= 64 
        # hps.img_crop_size = 108 
        hps.zsize = 1500
        hps.channels = 3
        hps.binary_reco_loss = False
        hps.vae_model = 'ConvResBlock32'
        hps.zclamp_min =-1 
        hps.zclamp = 1 
        hps.zround = -1

    return hps

if __name__ == "__main__":
    hps = get_params()
    exec(hps)

