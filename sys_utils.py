__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import os
import sys
import torch
import json
import numpy as np
import subprocess

def run_command(command):
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return '\n'.join([ '\t'+line.decode("utf-8").strip() for line in p.stdout.readlines()])

def ge_pkg_versions():
    import platform
    import pkg_resources
    import h5py
    import pickle
    import imageio
    import sklearn
    import tensorflow
    import PIL
    import  tensorboardX

    dep_versions = {}
    dep_versions['display'] = run_command('cat /proc/driver/nvidia/version')

    dep_versions['cuda'] = 'NA'
    cuda_home = '/usr/local/cuda/'
    if 'CUDA_HOME' in os.environ:
        cuda_home = os.environ['CUDA_HOME']

    cmd = cuda_home+'/version.txt'
    if os.path.isfile(cmd):
        dep_versions['cuda'] = run_command('cat '+cmd)

    dep_versions['cudnn'] = torch.backends.cudnn.version()
    dep_versions['platform'] = platform.platform()
    dep_versions['python'] = sys.version_info[:3]
    dep_versions['torch'] = torch.__version__
    dep_versions['torchvision'] = pkg_resources.get_distribution("torchvision").version
    dep_versions['tensorboardX'] =  tensorboardX.__version__
    dep_versions['tensorflow'] = tensorflow.__version__
    dep_versions['numpy'] = np.__version__
    dep_versions['h5py'] = h5py.__version__
    dep_versions['json'] = json.__version__
    # dep_versions['ortools'] = ortools.__version__
    dep_versions['pickle'] = pickle.format_version
    dep_versions['imageio'] = imageio.__version__
    dep_versions['sklearn'] = sklearn.__version__
    dep_versions['PIL'] = PIL.__version__

    # dep_versions['OpenCV'] = 'NA'
    # if 'cv2' in sys.modules:
    #     dep_versions['OpenCV'] = cv2.__version__
    return dep_versions

def print_pkg_versions():
    print("Packages & system versions:")
    print("----------------------------------------------------------------------")
    versions = ge_pkg_versions()
    for key, val in versions.items():
        print(key,": ",val)
    print("")
    return

def tohms(s):
    m,s=divmod(s,60)
    h,m=divmod(m,60)
    d,h=divmod(h,24)
    # return d,h,m,s
    time_str = "{:>02d}:{:>02}:{:>02d}".format(int(h), int(m), int(s))
    return time_str

def net_info(net):
    print(net)
    print('Parameters and size:')
    # for name, param in net.named_parameters():
        # print('{}: {}'.format(name, list(param.size())))

    num_params = sum([param.nelement() for param in net.parameters()])
    print('\nTotal number of parameters: {}\n'.format(num_params))
    return

#=================================================================================
if __name__ == "__main__":
    print_pkg_versions()

