# Authors: Edouard Oyallon
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna


#__all__ = ['Scattering2D']
#from try_cur import currr,fftt
#import pyct as ct
import torch
from scattering2d.backend import cdgmm, Modulus, SubsampleFourier, fft, Pad, unpad, convert_filters
from scattering2d.filter_bank import filter_bank
from scattering2d.utils import compute_padding
import scipy.io



class Scattering2D(object):
    """Main module implementing the scattering transform in 2D.
    The scattering transform computes two wavelet transform followed
    by modulus non-linearity.
    It can be summarized as::

        S_J x = [S_J^0 x, S_J^1 x, S_J^2 x]

    where::

        S_J^0 x = x * phi_J
        S_J^1 x = [|x * psi^1_lambda| * phi_J]_lambda
        S_J^2 x = [||x * psi^1_lambda| * psi^2_mu| * phi_J]_{lambda, mu}

    where * denotes the convolution (in space), phi_J is a low pass
    filter, psi^1_lambda is a family of band pass
    filters and psi^2_mu is another family of band pass filters.
    Only Morlet filters are used in this implementation.
    Convolutions are efficiently performed in the Fourier domain
    with this implementation.

    Example
    -------
        # 1) Define a Scattering object as:
        s = Scattering2D(J, shape=(M, N))
        #    where (M, N) are the image sizes and 2**J the scale of the scattering
        # 2) Forward on an input Tensor x of shape B x M x N,
        #     where B is the batch size.
        result_s = s(x)

    Parameters
    ----------
    J : int
        logscale of the scattering
    shape : tuple of int
        spatial support (M, N) of the input
    L : int, optional
        number of angles used for the wavelet transform
    max_order : int, optional
        The maximum order of scattering coefficients to compute. Must be either
        `1` or `2`. Defaults to `2`.
    pre_pad : boolean, optional
        controls the padding: if set to False, a symmetric padding is applied
        on the signal. If set to true, the software will assume the signal was
        padded externally.

    Attributes
    ----------
    J : int
        logscale of the scattering
    shape : tuple of int
        spatial support (M, N) of the input
    L : int, optional
        number of angles used for the wavelet transform
    max_order : int, optional
        The maximum order of scattering coefficients to compute.
        Must be either equal to `1` or `2`. Defaults to `2`.
    pre_pad : boolean
        controls the padding
    Psi : dictionary
        containing the wavelets filters at all resolutions. See
        filter_bank.filter_bank for an exact description.
    Phi : dictionary
        containing the low-pass filters at all resolutions. See
        filter_bank.filter_bank for an exact description.
    M_padded, N_padded : int
         spatial support of the padded input

    Notes
    -----
    The design of the filters is optimized for the value L = 8

    pre_pad is particularly useful when doing crops of a bigger
     image because the padding is then extremely accurate. Defaults
     to False.

    """

    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False):
        self.J, self.L = J, L
        self.pre_pad = pre_pad
        self.max_order = max_order
        self.shape = shape
        # self.currr = currr####################
        if 2 ** J > shape[0] or 2 ** J > shape[1]:
            raise (RuntimeError('The smallest dimension should be larger than 2^J'))

        self.build()

    def build(self):
        self.M, self.N = self.shape
        self.modulus = Modulus()
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        self.pad = Pad([(self.N_padded - self.N) // 2, (self.N_padded - self.N + 1) // 2, (self.M_padded - self.M) // 2,
                        (self.M_padded - self.M + 1) // 2], [self.N, self.M], pre_pad=self.pre_pad)
        self.subsample_fourier = SubsampleFourier()

        # Create the filters
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L)
        self.Psi = convert_filters(filters['psi'])
        self.Phi = convert_filters([filters['phi'][j] for j in range(self.J)])

    def _apply(self, fn):
        """
            Mimics the behavior of the function _apply() of a nn.Module()
        """
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = fn(item2)
        self.Phi = [fn(v) for v in self.Phi]
        self.pad.padding_module._apply(fn)
        return self

    def cuda(self, device=None):
        """
            Mimics the behavior of the function cuda() of a nn.Module()
        """
        return self._apply(lambda t: t.cuda(device))

    def to(self, *args, **kwargs):
        """
            Mimics the behavior of the function to() of a nn.Module()
        """
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        def convert(t):
            return t.to(device, dtype if t.is_floating_point() else None, non_blocking)

        return self._apply(convert)

    def cpu(self):
        """
            Mimics the behavior of the function cpu() of a nn.Module()
        """
        return self._apply(lambda t: t.cpu())

    def forward(self, input):
        """Forward pass of the scattering.

        Parameters
        ----------
        input : tensor
            tensor with 3 dimensions :math:`(B, C, M, N)` where :math:`(B, C)` are arbitrary.
            :math:`B` typically is the batch size, whereas :math:`C` is the number of input channels.

        Returns
        -------
        S : tensor
            scattering of the input, a 4D tensor :math:`(B, C, D, Md, Nd)` where :math:`D` corresponds
            to a new channel dimension and :math:`(Md, Nd)` are downsampled sizes by a factor :math:`2^J`.

        """


        if not torch.is_tensor(input):
            raise (
                TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if len(input.shape) < 2:
            raise (RuntimeError('Input tensor must have at least two '
                                'dimensions'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if ((input.size(-1) != self.N or input.size(-2) != self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!' % (self.M, self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        #print ("innnnnnnnnnnnnnnnn",input.shape)(128, 3, 32, 32)
        batch_shape = input.shape[:-2]
        #print (batch_shape)(128,3)
        signal_shape = input.shape[-2:]
        #print (signal_shape)(32,32)

        input = input.reshape((-1, 1) + signal_shape)
        #print (input.shape)(384, 1, 32, 32)

        J = self.J#####J=2
        phi = self.Phi
        psi = self.Psi
        #print ("phiiiii",len(phi))   2
        #print ("psiiiiii", len(psi))   16
        subsample_fourier = self.subsample_fourier
        modulus = self.modulus
        pad = self.pad
        order0_size = 1
        order1_size = self.L * J##############16, L=8
        order2_size = self.L ** 2 * J * (J - 1) // 2##########64
        output_size = order0_size + order1_size##################17
        if self.max_order == 2:#################meipao

            output_size += order2_size

        S = input.new(input.size(0),
                      input.size(1),
                      output_size,
                      self.M_padded // (2 ** J) - 2,
                      self.N_padded // (2 ** J) - 2)
        #print (S.shape)(384, 1, 17, 8, 8)
        U_r = pad(input)
        #print(U_r.shape)#(384, 1, 40, 40, 2)

        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c
        #print (U_0_c.shape)(384, 1, 40, 40, 2)
        # First low pass filter
        U_1_c = subsample_fourier(cdgmm(U_0_c, phi[0]), k=2 ** J)##(384, 1, 10, 10, 2)
        
        #aaa = U_1_c.detach().cpu()
        #aaa=aaa.float()
        #aaa=aaa.numpy()
        #size = (aaa.shape[0])*(aaa.shape[1])*(aaa.shape[2])*(aaa.shape[3]*(aaa.shape[4]))
        #aaa.reshape((1,size))
        #scipy.io.savemat('u1c.mat',{'aaa':aaa})
        ###############################################################################print ("lookkkkkk",U_1_c.shape)##(384, 1, 10, 10, 2)
        
        # print ("U1CCCCCCCCCCCCCCCCC",U_1_c)
        # tmpp = currr(U_0_c,(U_0_c.shape[0], 1, 10,10, 2))##################
        # print ("tmpppppppppppppp",tmpp)
        # U_1_c += currr(U_0_c,(U_0_c.shape[0], 1, 10,10, 2))##################
        U_J_r = fft(U_1_c, 'C2R')##(384, 1, 10, 10)
        # U_J_r = fftt(U_1_c, (U_0_c.shape[0], 1, 10,10))
        ################################################################################ print ("lookkkkkk",U_J_r.shape)##(384, 1, 10, 10)

        S[..., 0, :, :] = unpad(U_J_r)
        #print ("S[..., 0, :, :]",S[..., 0, :, :].shape)  (384, 1, 8, 8)
        n_order1 = 1
        n_order2 = 1 + order1_size

        for n1 in range(len(psi)):
            j1 = psi[n1]['j']
            #print ("psi",len(psi))  16
            #print ("j1",j1) 8 ge 0 8 ge 1
            U_1_c = cdgmm(U_0_c, psi[n1][0])
            if (j1 > 0):
                # tmp = U_1_c##################
                #print ("hahahahahah")
                U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
                # U_1_c += currr(tmp,(U_0_c.shape[0], 1, 20,20, 2))######################
            U_1_c = fft(U_1_c, 'C2C', inverse=True)
            U_1_c = fft(modulus(U_1_c), 'C2C')
            #print (U_1_c.shape)    8ge(384, 1, 40, 40, 2)   8ge(384, 1, 20, 20, 2)


            # Second low pass filter
            U_2_c = subsample_fourier(cdgmm(U_1_c, phi[j1]), k=2 ** (J - j1))
            U_J_r = fft(U_2_c, 'C2R')
            #print ("U_J_r",U_J_r.shape) (384, 1, 10, 10)
            S[..., n_order1, :, :] = unpad(U_J_r)
            #print ("S[..., n_order1, :, :]", S[..., n_order1, :, :].shape) (384, 1, 8, 8)
            n_order1 += 1

            if self.max_order == 2:
                #print("aaaaaaaaaaaaaaaaaaaaaaaaa")
                for n2 in range(len(psi)):
                    j2 = psi[n2]['j']
                    #print("j2",j2) 0 or 1
                    if (j1 < j2):
                        U_2_c = subsample_fourier(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2 - j1))
                        U_2_c = fft(U_2_c, 'C2C', inverse=True)
                        U_2_c = fft(modulus(U_2_c), 'C2C')

                        # Third low pass filter
                        U_2_c = subsample_fourier(cdgmm(U_2_c, phi[j2]), k=2 ** (J - j2))
                        U_J_r = fft(U_2_c, 'C2R')
                        #print ("U_JJ_r", U_J_r.shape) (384, 1, 10, 10)
                        S[..., n_order2, :, :] = unpad(U_J_r)
                        #print ("S[..., n_order2, :, :]", S[..., n_order2, :, :].shape) (384, 1, 8, 8)
                        n_order2 += 1

        scattering_shape = S.shape[-3:]
        #print("scattering_shape",scattering_shape)  (81, 8, 8)
        S = S.reshape(batch_shape + scattering_shape)
        #print ("s",S.shape) (128, 3, 81, 8, 8)
        return S

    def __call__(self, input):
        return self.forward(input)


