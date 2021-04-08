from __future__ import absolute_import, division, print_function
__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import random
import warnings
warnings.filterwarnings('ignore')
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
import glob
import numpy as np
import fid
import imageio
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics.pairwise import polynomial_kernel

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from image_utils import psnr, ssim

rnd_seed = 12345
random.seed(rnd_seed)
np.random.seed(rnd_seed)
tf.compat.v2.random.set_seed(rnd_seed)
tf.random.set_random_seed(rnd_seed)


BATCH_SIZE = 100

def eval_init():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    # config = ConfigProto(device_count={'GPU': 1})
    config = ConfigProto()
    # config = ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)



def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est

def precalc(data_path, output_path):
    print("CALCULATING THE GT STATS....")
    # data_path = 'reconstructed_test/eval' # set path to training set images
    # output_path = data_path+'/fid_stats.npz' # path for where to store the statistics
    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
    print("ok")

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    image_list = glob.glob(os.path.join(data_path, '*.jpg'))
    if len(image_list) == 0:
        print("No images in directory ", data_path)
        return

    images = np.array([imageio.imread(str(fn),as_gray=False, pilmode="RGB").astype(np.float32) for fn in image_list])
    print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma, acts = fid.calculate_activation_statistics(images, sess, batch_size=BATCH_SIZE)
        np.savez_compressed(output_path, mu=mu, sigma=sigma, activations=acts)
    print("finished")

def fid_imgs(cfg):
    print("CALCULATING FID/KID scores")
    rnd_seed = 12345
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    tf.compat.v2.random.set_seed(rnd_seed)
    tf.random.set_random_seed(rnd_seed)
    inception_path = fid.check_or_download_inception(None) # download inception network

    # load precalculated training set statistics
    print("Loading stats from:", cfg.stats_filename, '  ...', end='')
    f = np.load(cfg.stats_filename)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]

    activations_ref = None
    if 'activations' in f:
        activations_ref = f['activations']
        print(" reference activations #:", activations_ref.shape[0])
        
    f.close()
    print("done")

    fid_epoch = 0
    epoch_info_file = cfg.exp_path+'/fid-epoch.txt'
    if os.path.isfile(epoch_info_file):
        fid_epoch = open(epoch_info_file, 'rt').read()
    else:
        print("ERROR: couldnot find file:", epoch_info_file)

    best_fid_file = cfg.exp_path+'/fid-best.txt'
    best_fid = 1e10
    if os.path.isfile(best_fid_file):
        best_fid = float(open(best_fid_file, 'rt').read())
        print("Best FID: "+str(best_fid))

    pr = None
    pr_file = cfg.exp_path+'/pr.txt'
    if os.path.isfile(pr_file):
        pr = open(pr_file).read()
        print("PR: "+str(pr))

    rec = []
    rec.append(fid_epoch)
    rec.append('nref:'+str(activations_ref.shape[0]))

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    dirs = cfg.image_path.split(',')
    first_fid = None
    for dir in dirs:
        print("Working on:",dir)
        test_name = dir.split('/')[-1]
        rec.append(test_name)
        # loads all images into memory (this might require a lot of RAM!)
        image_list = glob.glob(os.path.join(dir, '*.jpg'))
        image_list = image_list + glob.glob(os.path.join(dir, '*.png'))
        image_list.sort()
        print("Loading images:", len(image_list), '  ...', end='')
        images = np.array([imageio.imread(str(fn),as_gray=False, pilmode="RGB").astype(np.float32) for fn in image_list])
        print("done")

        print("Extracting features ", end='')
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            mu_gen, sigma_gen, activations = fid.calculate_activation_statistics(images, sess, batch_size=BATCH_SIZE)
        print("Extracted activations:", activations.shape[0])
        rec.append('ntest:'+str(activations.shape[0]))

        if cfg.fid:
            # Calculate FID
            print("Calculating FID.....")
            fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            rec.append('fid:'+str(fid_value))
            if first_fid is None:
                first_fid = fid_value
            
            if best_fid > first_fid and fid_epoch != 0:
                epoch = int(fid_epoch.split(' ')[0].split(':')[1])
                print("Storing best FID model. Epoch: "+str(epoch)+"  Current FID: "+str(best_fid)+" new: "+str(first_fid))
                best_fid = first_fid
                # Store best fid & weights
                with open(best_fid_file, 'wt') as f:
                    f.write(str(first_fid))
                model_file = cfg.exp_path+'/models/weights-'+str(epoch)+'.cp'
                backup_model_file = cfg.exp_path+'/models/'+str(epoch)+'.cp'
                os.system('cp ' + model_file + '  '+ backup_model_file)


        if cfg.kid:
            # Calculate KID
            # Parameters:
            print("Calculating KID...")
            mmd_degree=3
            mmd_gamma=None
            mmd_coef0=1
            mmd_var = False
            mmd_subsets=100
            mmd_subset_size=1000

            ret = polynomial_mmd_averages(
                activations, activations_ref, degree=mmd_degree, gamma=mmd_gamma,
                coef0=mmd_coef0, ret_var=mmd_var,
                n_subsets=mmd_subsets, subset_size=mmd_subset_size)

            if mmd_var:
                mmd2s, vars = ret
            else:
                mmd2s = ret
            
            kid_value = mmd2s.mean()
            kid_value_std = mmd2s.std()
            rec.append('kid_mean:'+str(kid_value))
            rec.append('kid_std:'+str(kid_value_std))


        if  cfg.psnr and test_name == 'reco':
            image_list = glob.glob(os.path.join(cfg.stats_path, '*.jpg'))
            image_list.sort()
            if len(image_list) == 0:
                print("No images in directory ", cfg.stats_path)
                return

            images_gt = np.array([imageio.imread(str(fn),as_gray=False, pilmode="RGB").astype(np.float32) for fn in image_list])
            print("%d images found and loaded" % len(images_gt))
            print("Calculating PSNR...")
            psnr_val = psnr(images_gt, images)
            print("Calculating SSIM...")
            ssim_val = ssim(images_gt, images)

            print('PSNR:', psnr_val, 'SSIM:', ssim_val)
            rec.append('psnr:'+str(psnr_val))
            rec.append('ssim:'+str(ssim_val))

        print(' '.join(rec))

    if pr is not None:
        rec.append(pr)

    print(' '.join(rec))

    # Write out results
    with open(cfg.exp_path+'/results.txt', 'a+') as f:
        f.write(' '.join(rec)+'\n')

    return first_fid
    
class EvalConfig:
    def __init__(self):
        return

#------------------------------------------------------------------
if __name__ == "__main__":
    image_path = 'generated/samples/' # set path to some generated images
    stats_path = 'reconstructed_test/eval' # set path to training set images
    stats_path = None

    for arg in sys.argv:
        toks = arg.split('=')
        if toks[0] =='s':
            stats_path=toks[1]
        if toks[0] =='t':
            image_path=toks[1]

    if stats_path is None:
        # Print help
        print("Help")
        print(sys.argv[0], " s=<path_to_GT_images>  t=<path_to_test_images>")
        print("\t File fid_stats.npz with the GT stats will be created in the dir with GT iamges")
        sys.exit(0)

    print('stats_path:', stats_path)
    print('image_path:', image_path)
    
    stats_filename = stats_path+'/fid_stats.npz'

    cfg = EvalConfig()
    cfg.exp_path = '.'
    cfg.image_path  = image_path
    cfg.stats_path = stats_path
    cfg.stats_filename = stats_filename
    cfg.kid = True
    cfg.fid = True
    cfg.psnr = False

    eval_init()
    # eval_init()
    # eval_init()

    if not os.path.isfile(stats_filename):
        precalc(stats_path, stats_filename)
        # sys.exit(0)
 
    fid_imgs(cfg)
    print("==============================================================")
