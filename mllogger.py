__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import os
import sys
import numpy as np
import time
import glob
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import imageio
import skimage

from parameters import Params
from sys_utils import tohms
from image_utils import save_image


#========================================================================================
class TeePipe(object):
    #source: https://stackoverflow.com/q/616645
    def __init__(self, filename="Red.Wood", mode="a", buff=0):
        self.stdout = sys.stdout
        # self.file = open(filename, mode, buff)
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None


# ========================================================================================
class MLLogger():
    def __init__(self, hps): 
        self.hps = hps
        self.logf = None


        self.im_size = hps.img_size
        self.epoch_num = hps.epochs_max  # Total number of epochs
        self.iter_num = {} # Iterations per epoch
        # self.iter_epoch = 0 
        self.batch_size = hps.batch_size

        self.data = []
        self.dkeys_id=['ts', 'epoch', 'iter', 'stage']  # Key lookup by ID
        self.dkeys = {}  # ID lookup by key
        self.m_first = {}  # stage_name -> position of first record
        self.m_last = {}  # stage_name -> position of last record
        self.start_time = None  # blobal start timestamp
        self.iter_global = 0  # Total iteration since the begining of the training

        self.print_header = True
        self.data_format_changed = True
        self.last_report_pos={}   # stage_name -> Position in self.data of the last report

        # Tensorboard 
        self.writer = None
        self.log_id = None
        return

    def load_config(self):
        logdir = self.exp_path+'/log/'
        cfg_filename = os.path.join(logdir, 'cfg-'+str(self.log_id-1)+'-*.json') 
        cfg_files = glob.glob(cfg_filename)
        cfg_files.sort(reverse=True)
        if len(cfg_files) == 0 or not os.path.isfile(cfg_files[0]):
            return None

        p = Params()
        if not p.load(cfg_files[0]):
            return None
        return p 

    def save_config(self, epoch=None):
        logdir = self.exp_path+'/log/'
        cfg_filename = os.path.join(logdir, 'cfg-'+str(self.log_id)+'-'+str(epoch)+'.json') 
        self.hps.save(cfg_filename)
        return

    def open_experiment(self, experiment_name='m1'):
        """
        Creates sub-directory structure
        - Creates new log file
        - 
        """
        self.experiment_name = experiment_name
        self.exp_path = os.path.join(experiment_name)
        os.makedirs(self.exp_path, exist_ok=True)
        
        if not self.hps.eval and self.experiment_name != '.':
            # Backup source code & configs
            os.system('cp *.py ' + self.exp_path+ '/')

        logdir = self.exp_path+'/tblog/'
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir)

        logdir = self.exp_path+'/log/'
        os.makedirs(logdir, exist_ok=True)
 
        self.model_path =os.path.join(self.exp_path, 'models') 
        os.makedirs(self.model_path, exist_ok=True)

        # Create new log files
        prefix = 'eval-' if self.hps.eval else 'train-' 
        log_id = 0
        while True:
            log_filename = prefix+'log-'+str(log_id)+'.txt'
            log_path = os.path.join(logdir, log_filename)
            if not os.path.isfile(log_path):
                break
            log_id += 1

        if self.hps.log_stdout:
            stdout_log_filename = prefix+'stdout-'+str(log_id)+'.txt'
            stdout_log_filename = os.path.join(logdir, stdout_log_filename)
            self.stdout_logger = TeePipe(stdout_log_filename)

        print("Creating new log file:",log_path)
        self.logf = open(log_path, 'wt')
        self.log_id = log_id
        return


    def set_samples_num(self, stage_name, samples_num):
        self.iter_num[stage_name] = self.hps.batch_size * int(np.floor(samples_num / self.hps.batch_size))


    def start_epoch(self, stage_name, epoch):
        """
        Creates a null record with a current timestamp
        """
        if self.start_time is None:
            self.start_time = time.time()

        # Stored the position of the first epoch record
        # There can be one start per stage
        self.m_first[stage_name] = len(self.data)
        self.m_last[stage_name] = len(self.data)
        self.last_report_pos[stage_name] = len(self.data)

        rec = [0]*len(self.dkeys_id)
        rec[0] = time.time() - self.start_time
        rec[1] = epoch
        rec[2] = self.iter_global
        rec[3] = stage_name
        self.data.append(rec)
        self.print_header = True
        return


    def log_loss(self, epoch, iter, losses, stage_name):
        """
        Args:
        epoch (int):    current epoch starting from 0
        iter (int):     sample iteration within the epoch
        stage_name (str):   'train', 'val', 'test'
        losses (dict):  dictionary of   loss_name->loss_val
        """

        if iter is not None:
            self.iter_global = iter

        # Collect new value keys
        for key, val in losses.items():
            if key not in self.dkeys_id:
                # Add new key=val
                self.dkeys_id.append(key)
                self.data_format_changed = True

        # Update the key-index lookup table
        if self.data_format_changed:
            self.dkeys = {}
            for i, key in enumerate(self.dkeys_id):
                self.dkeys[key] = i

        # Store new data
        rec = [0]*len(self.dkeys_id)
        rec[0] = time.time() - self.start_time
        rec[1] = epoch
        rec[2] = self.iter_global  # Global iteration
        rec[3] = stage_name

        # Generate tensorboar record
        tboard_losses = {}
        for key, val in losses.items():
            id = self.dkeys[key]
            rec[id] = val
            key = stage_name+'_'+key
            tboard_losses[key] = val

        self.data.append(rec)


        # Append log to the file
        if self.logf is not None:
            if self.data_format_changed:
                # Insert data format header
                header_str = [str(v) for v in self.dkeys_id]
                self.logf.write('\n'+' '.join(header_str)+'\n')

            line = [str(v) for v in rec]
            self.logf.write(' '.join(line)+'\n')
            self.logf.flush()

        # Update tensorboard
        # {'d_loss': d_loss, 'grad_penalty': grad_penalty}
        self.writer.add_scalars('losses', tboard_losses, self.iter_global)

        self.m_last[stage_name] = len(self.data)-1
        self.data_format_changed= False
        return

    def print_table(self, name, data, header=None):
        """
        max_iter = self.iter_num*self.epoch_num
        epoch_str = str(rec[-1][1])+" ("+str(int(done))+"%)"
        header = ['T', 'e('+str(self.epoch_num)+')', 'iter('+str(max_iter//1000)+'k)', 'batch (ms)']
        data = [[rec[-1][3], epoch_str, str(last_iter), batch_took_avg*1000.0]]
        """

        # Print table
        table_width = 0
        if header is not None:
            self.col_width = []
            line = ""
            for i, hv in enumerate(header):
                line += '{:<{c0}}'.format(hv, c0=len(hv))
                self.col_width.append(len(hv))

            print('')
            if name is not None:
                print(name)
            print(line)

            head_len = len(line)
            print('-'*head_len )
            table_width = head_len

        # Print data
        for r, rv in enumerate(data):
            line = ""
            for c, cv in enumerate(rv):
                line += '{:<{c0}}'.format(cv, c0=self.col_width[c])
            print(line, flush=True)
            if len(line) > table_width:
                table_width = len(line)

        return table_width
             
    def get_avg(self, begin, end, cols=[]):
        rec = self.data[begin:end]

        # Get the max number of stored value in this run
        mx = 0
        for val in rec:
            if len(val)>mx: mx = len(val)

        # Create numpy vector for the averages
        rec = np.asarray([x+[0]*(mx-len(x)) for x in rec], dtype=np.object )

        # Get only the records with loss values 
        rec_avg  = rec.copy()
        rec_avg[:,:4] = 0
        rec_avg = rec_avg.astype(np.float)
        rec_avg = rec_avg.mean(0)
        return rec_avg


    def print_batch_stat(self, stage_name='t'):
        last_epoch_pos = self.m_last.get(stage_name, 0)
        last_report_pos = self.last_report_pos.get(stage_name, 0)

        if last_report_pos == last_epoch_pos:
            # Already reported
            return

        # Get averages since the last report
        rec_avg = self.get_avg(last_report_pos+1, last_epoch_pos+1)
        rec_last = self.data[last_epoch_pos]
        time_now, last_epoch, last_iter, last_stage_name = rec_last[:4]
        iter = last_iter - self.data[self.m_first[stage_name]][2]
        done = round(100*iter/self.iter_num.get(stage_name), 2) if stage_name in self.iter_num else 0
        batch_took_avg = float(time_now) - float(self.data[last_report_pos+1][0])
        if self.batch_size is not None:
            batch_took_avg /= self.batch_size

        self.last_report_pos[stage_name] = last_epoch_pos

        # Print table
        header =  None
        if self.print_header:
            max_iter = self.iter_num.get(stage_name, 0)*self.epoch_num
            header = ['Time      ',
                      'E('+str(self.epoch_num)+')        ',
                      'Iter('+str(max_iter//1000)+'k)   ',
                      'Batch (ms)   ']
            for key in  self.dkeys_id[4:]:
                header.append(key+' '*(15-len(key)))
            self.print_header=False 

        data = [tohms(time_now), str(last_epoch)+' ('+str(done)+'%)', str(last_iter), round(batch_took_avg*1000.0, 3)]
        for key in self.dkeys_id[4:]:
            data.append(round(rec_avg[self.dkeys[key]], 4))
        table_width = self.print_table(last_stage_name, [data], header)
        return

    def print_epoch_stat(self, stage_name, **kwargs):
        """
        Batch train log format
        Epoch train log format
        Test log format
        """
        first_epoch_pos = self.m_first.get(stage_name, 0)
        last_epoch_pos = self.m_last.get(stage_name, 0)
        rec_avg = self.get_avg(first_epoch_pos+1, last_epoch_pos+1)
        rec_last = self.data[last_epoch_pos]
        time_now, last_epoch, last_iter, last_stage_name = rec_last[:4]
        epoch_took = tohms(time_now - self.data[first_epoch_pos][0])

        # Print table
        max_iter = self.iter_num*self.epoch_num
        header = ['Time      ',
                  'E('+str(self.epoch_num)+')                  ',
                  'Iter('+str(max_iter//1000)+'k)        ',
                  'Epoch (H:M:S)      ']
        for key in  self.dkeys_id[4:]:
            header.append(key)

        data = [tohms(time_now), str(last_epoch), str(last_iter), epoch_took]
        for key in self.dkeys_id[4:]:
            data.append(round(rec_avg[self.dkeys[key]], 4))

        table_width = self.print_table(last_stage_name, [data], header)
        print("-"*table_width)
        return
    
    def log_images(self, x, epoch, name_suffix, name, channels=3, nrow=8):

        img_path = os.path.join(self.experiment_name, name)
        os.makedirs(img_path, exist_ok=True)
        
        img_size = self.im_size
        if img_size < 1:
            img_size2 = x.nelement() / x.size(0) / channels
            img_size = int(np.sqrt(img_size2))

        x = x.view(-1, channels, img_size, img_size)   # * 0.5 + 0.5

        grid = save_image(x, 
                   img_path+'/sample_' + str(epoch) + "_" + str(name_suffix) + '.jpg',
                   nrow = nrow, normalize=True, scale_each=True)
        
        img_grid = make_grid(x, normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image(name, img_grid , self.iter_global)
        return


    def _merge(self, images, size, labels=[], strike=[]):
        h, w = images.shape[1], images.shape[2]

        resize_factor=1.0
        h_ = int(h * resize_factor)
        w_ = int(w * resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = skimage.transform.resize(image, output_shape=(w_, h_))

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_
            if len(labels) == len(images):
                if labels[idx] == 1:
                    img[j * h_:j * h_ + 2, i * w_:i * w_ + w_-4] = np.ones((2, w_-4))

            if len(strike) == len(images):
                if strike[idx] == 1:
                    img[j * h_+h_//2:j * h_ + h_//2+1, i * w_:i * w_ + w_-4] = np.ones((1, w_-4))
        return img


    def save_images(self, images, img_size=(28,28), labels=[], strike=[], name='result.jpg'):
        n_img_y = 16
        n_img_x = 32
        images = images.reshape(n_img_x * n_img_y, img_size[0], img_size[1])
        imageio.imsave(name, self._merge(images, [n_img_y, n_img_x], labels, strike))


#=================================================================================
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")



