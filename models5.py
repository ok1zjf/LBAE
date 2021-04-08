__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import torch
import torch.nn as nn

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

def weight_init(model, mean=0, std=0.02):
    for m in model._modules:
        normal_init(model._modules[m], mean, std)
    return

#===========================================================================================
class QuantizerFunc(torch.autograd.Function):
    @staticmethod
    def forward(self, input, npoints=4, dropout=0):
        # self.save_for_backward(input)
        # self.constant = npoints
        if npoints < 0:
            x = torch.sign(input)
            x[x==0] = 1
            return x

        scale = 10**npoints
        input = input * scale
        input = torch.round(input)
        input = input / scale
        return input

    @staticmethod
    def backward(self, grad_output):
        # input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # grad_input[:] = 1
        return grad_input, None


class Quantizer(nn.Module):
    def __init__(self, npoints=3):
        super().__init__()
        self.npoints = npoints
        self.quant = QuantizerFunc.apply

    def forward(self,x):
        x = self.quant(x, self.npoints)
        return x

#===========================================================================================
class BinDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.mask = None

    def forward(self, x):
        if self.training:
            if self.mask is None:
                po = int(self.p * x.size(1))
                self.mask = 1
                self.bdist = torch.distributions.Bernoulli(self.p)

            m = self.bdist.sample(x.size())*-2+1 
            x = x*m.cuda()
            return x
        return x

#===========================================================================================
class EncConvResBlock32(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps=hps
        channels = hps.channels
        bias = False 
        c = 64

        inc = c
        self.ec0 = nn.Conv2d(channels, inc, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn0 = nn.BatchNorm2d(inc)
        self.ec1 = nn.Conv2d(inc, c, kernel_size=4, stride=2, padding=1, bias=bias)

        self.bn1 = nn.BatchNorm2d(c)
        self.b11 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn11 = nn.BatchNorm2d(c)
        self.b12 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn12 = nn.BatchNorm2d(c)

        c = c*2
        self.ec2 = nn.Conv2d(c//2, c, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(c)
        self.b21 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn21 = nn.BatchNorm2d(c)
        self.b22 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn22 = nn.BatchNorm2d(c)

        c_out = c*2
        if self.hps.img_size == 64:
            self.ec3 = nn.Conv2d(c, c_out, kernel_size=4, stride=2, padding=1, bias=bias)
            self.bn3 = nn.BatchNorm2d(c_out)
            self.b31 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=bias)
            self.bn31 = nn.BatchNorm2d(c_out)
            self.b32 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=bias)
            self.bn32 = nn.BatchNorm2d(c_out)
            c = c_out
        
        self.ec4 = nn.Conv2d(c, c_out, kernel_size=4, stride=2, padding=1, bias=False)

        fmres = 4*4 
        in_size =self.ec4.out_channels*fmres

        if hps.vae:
            self.l1 = nn.Linear(in_size, self.hps.zsize)
            self.l2 = nn.Linear(in_size, self.hps.zsize)
        else:
            self.l0l = nn.Linear(in_size, self.hps.zsize)
            self.quant = QuantizerFunc.apply

        self.act = nn.LeakyReLU(0.02)
        self.drop = None

    def forward(self, x):
        x = self.ec0(x)
        x = self.bn0(x)
        x = self.act(x)

        x = self.ec1(x)
        x = self.bn1(x)
        y = x
        x = self.act(x)
        x = self.b11(x)
        x = self.bn11(x)
        x = self.act(x)
        x = self.b12(x)
        x = self.bn12(x)
        x = self.act(x+y)
        
        x = self.ec2(x)
        x = self.bn2(x)
        y = x
        x = self.act(x)
        x = self.b21(x)
        x = self.bn21(x)
        x = self.act(x)
        x = self.b22(x)
        x = self.bn22(x)
        x = self.act(x+y)

        if self.hps.img_size == 64:
            x = self.ec3(x)
            x = self.bn3(x)
            y = x
            x = self.act(x)
            x = self.b31(x)
            x = self.bn31(x)
            x = self.act(x)
            x = self.b32(x)
            x = self.bn32(x)
            x = self.act(x+y)

        x = self.ec4(x)
        x = x.view(x.size(0), -1)

        if not self.hps.vae:
            # QAE output
            xe = None

            x = self.l0l(x)
            x = torch.tanh(x)

            xq = self.quant(x, self.hps.zround)
            err_quant = torch.abs(x - xq)
            x = xq

            xe = x if xe is None else xe
            diff = ((x+xe) == 0).sum(1) 
            return x, None, xe, diff, err_quant.sum()/(x.size(0) * x.size(1))
        else:
            # VAE output
            mu = self.l1(x)
            logvar = self.l2(x)
            return mu, logvar, mu, None,None

#===========================================================================================
class GenConvResBlock32(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        channels = hps.channels

        bias = False
        c = inch = 128

        if self.hps.dataset == 'mnist':
           inch = 1

        self.in_channels = inch
        if self.hps.img_size == 64:
            c = inch = 256
            self.dc1 = nn.ConvTranspose2d(inch, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
            self.bn1 = nn.BatchNorm2d(c)
            self.b11 = nn.ConvTranspose2d(c, c,kernel_size=3, stride=1, padding=1, bias=bias)
            self.bn11 = nn.BatchNorm2d(c)
            self.b12 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
            self.bn12 = nn.BatchNorm2d(c)
            inch = c
            c = c//2

        self.dc2 = nn.ConvTranspose2d(inch, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(c)
        self.b21 = nn.ConvTranspose2d(c, c,kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn21 = nn.BatchNorm2d(c)
        self.b22 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn22 = nn.BatchNorm2d(c)

        c = c//2
        self.dc3 = nn.ConvTranspose2d(c*2, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
        self.bn3 = nn.BatchNorm2d(c)
        self.b31 = nn.ConvTranspose2d(c, c,kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn31 = nn.BatchNorm2d(c)
        self.b32 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn32 = nn.BatchNorm2d(c)

        self.dc4 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(c)
        self.dc5 = nn.ConvTranspose2d(c, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.act = nn.LeakyReLU(0.02)
        self.quant = QuantizerFunc.apply

        if self.hps.img_size == 64:
            self.in_channels = self.dc1.in_channels
        else:
            self.in_channels = self.dc2.in_channels

        self.fmres = 4 
        out_size = self.in_channels*self.fmres*self.fmres
        bias = True
        self.l1l=nn.Linear(self.hps.zsize, out_size, bias=bias)
        

    def forward(self, x, sw=None):
        x = x.view(x.size(0), -1)

        x = self.l1l(x)
        x = x.view(x.size(0), self.in_channels,self.fmres,self.fmres)

        if self.hps.img_size == 64:
            x = self.dc1(x)
            x = self.bn1(x)
            y = x
            x = self.act(x)
            x = self.b11(x)
            x = self.bn11(x)
            x = self.act(x)
            x = self.b12(x)
            x = self.bn12(x)
            x = self.act(x+y)

        x = self.dc2(x)
        x = self.bn2(x)
        y = x
        x = self.act(x)
        x = self.b21(x)
        x = self.bn21(x)
        x = self.act(x)
        x = self.b22(x)
        x = self.bn22(x)
        x = self.act(x+y)

        x = self.dc3(x)
        x = self.bn3(x)
        y = x
        x = self.act(x)
        x = self.b31(x)
        x = self.bn31(x)
        x = self.act(x)
        x = self.b32(x)
        x = self.bn32(x)
        x = self.act(x+y)

        x = self.dc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dc5(x)

        x = torch.sigmoid(x)
        return x

#=================================================================================
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")




