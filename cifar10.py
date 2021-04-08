from mllogger import *
from lbae import *

def main():
    # Default parameters
    hps = Params()
    hps.cuda_device = 0
    hps.zsize = 100
    hps.batch_size = 64
    hps.batch_size_test = 64
    hps.epochs_max = 3900 
    hps.lr[0] = 1e-3
    hps.img_size = 28
    hps.channels = 1
    hps.model_conv = False
    hps.shared_weights = False
    hps.workers = 10
    hps.print_every_batch = 100
    hps.keep_last_models = 10
    hps.binary_reco_loss = False
    hps.corrupt_method_test = 'blank'
    hps.corrupt_args_test = []
    hps.corrupt_method = 'blank'
    hps.corrupt_args = []
    hps.vae_model = None

    hps.parallel = False
    hps.vae = False
    hps.sample_method = 'kde'
    hps.kde_bw = 0.1
    hps.gen_knn = False
    hps.znoise_std = 0
    hps.l2 = 0
    hps.zdrop = 0
    hps.zclamp_min =-1 
    hps.zclamp = 1 
    hps.zround = -1
    hps.gen_imgs = 512
    hps.keep_last_models = 10
    hps.zsize_cont_enc = 0
    hps.zsize_cont = 0
    hps.set_attr = -1
    hps.dataset = None

    hps.cfg = 'cifar10_bae'
    # hps.cfg = 'cifar10_vae'

#=====================================================
    if hps.cfg == 'cifar10_vae':
        hps.dataset = 'cifar10'
        hps.exp_suffix = 'm173'
        hps.vae = True
        hps.lr[0] = 1e-3
        hps.channels = 3
        hps.img_size= 32
        hps.keep_last_models = 10
        hps.kl_weight = 128
        hps.zsize = 128
        hps.batch_size = 512
        hps.batch_size_test = 1024
        hps.sample_method = 'random'
        hps.gen_imgs = 5000
        hps.binary_reco_loss = False
        hps.vae_model = 'ConvResBlock32'

#=====================================================
    if 1 and hps.cfg == 'cifar10_bae':
        hps.exp_suffix = 'm171-3'
        hps.dataset = 'cifar10'
        # hps.parallel = True
        hps.epochs_max = 3000 
        hps.channels = 3
        hps.img_size= 32
        hps.batch_size = 512
        hps.batch_size_test = 1024
        hps.sample_method = 'cov'
        hps.gen_imgs = 5000
        hps.interpolate_steps = 10
        hps.vae_model = 'ConvResBlock32'
        hps.lr[0] = 1e-3
        hps.zsize = 600

#=====================================================
    if 0 and hps.cfg == 'cifar10_bae':
        # TRAIN TEST !!!!

        hps.exp_suffix = 'm171-9'
        hps.dataset = 'cifar10'
        # hps.parallel = True
        hps.epochs_max = 3000 
        hps.channels = 3
        hps.img_size= 32
        hps.batch_size = 512
        hps.batch_size_test = 1024
        hps.sample_method = 'cov'
        hps.gen_imgs = 5000
        hps.interpolate_steps = 10
        hps.vae_model = 'ConvResBlock32'
        # hps.vae_model = 'Net'
        # hps.vae_model = 'NetX'
        # hps.vae_model = '3ConvNet'
        # hps.l2 = 1e-4
        hps.lr[0] = 1e-3
        # hps.lr[0] = 1e-4
        hps.zsize = 600
        # hps.zsize_cont_enc = 0
        # hps.zsize_cont = 2048
        # hps.zdrop = 0.05

    exec(hps)

main()
