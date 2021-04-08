from mllogger import *
from lbae import *

def main():
    hps = Params()
    hps.cuda_device = 0
    hps.zsize = 100
    hps.batch_size = 64
    hps.batch_size_test = 64
    hps.epochs_max = 1000 
    hps.lr[0] = 1e-3
    hps.img_size = 28
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

    hps.parallel = False
    hps.vae = False
    hps.sample_method = 'kde'
    hps.znoise_std = 0
    hps.l2 = 0
    hps.gen_imgs = 512
    hps.kl_weight = 1.0
    hps.zdrop = 0
    hps.zclamp_min =-1 
    hps.zclamp = 1 
    hps.zround = -1
    hps.zsize_cont = 0 
    hps.zsize_cont_enc = 0 
    hps.set_attr = -1

    hps.dataset = None
    hps.cfg = 'mnist_bae'


#============================================================
    if hps.cfg == 'mnist_vae':
        hps.exp_suffix = 'm64'
        hps.dataset = 'mnist'
        # hps.parallel = True
        hps.vae = True
        hps.kl_weight = 10.0
        hps.l2 = 1e-4
        hps.lr[0] = 1e-3
        hps.lr[0] = 1e-4
        hps.channels = 1
        hps.img_size= 32
        hps.zsize = 20
        hps.batch_size = 512
        hps.batch_size_test = 512
        hps.sample_method = 'random'
        hps.gen_imgs = 512
        hps.binary_reco_loss = True
        hps.vae_model = 'ConvResBlock32'
        hps.zclamp_min =-1 
        hps.zclamp = 1 
        hps.zround = -1

    if hps.cfg == 'mnist_bae':
        hps.exp_suffix = 'm169'
        hps.dataset = 'mnist'
        hps.epochs_max = 5000 
        hps.binary_reco_loss = True
        hps.channels = 1
        hps.img_size= 32
        hps.batch_size = 512
        hps.batch_size_test = 1024
        hps.sample_method = 'cov'
        hps.gen_imgs = 10000
        hps.interpolate_steps = 10
        hps.vae_model = 'ConvResBlock32'
        hps.lr[0] = 1e-4
        hps.zsize = 200

    exec(hps)
main()
