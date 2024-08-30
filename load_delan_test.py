import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill as pickle
import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk
import warnings
import numpy as np
import numpy.random as random
import torch
import jax.numpy as jnp
import argparse
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

import matplotlib as mp

# try:
#     mp.use("Qt5Agg")
#     mp.rc('text', usetex=True)
#     mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# except ImportError:
#     pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


class ReplayMemory(object):
    def __init__(self, maximum_number_of_samples, minibatch_size, dim):

        # General Parameters:
        self._max_samples = maximum_number_of_samples
        self._minibatch_size = minibatch_size
        self._dim = dim
        self._data_idx = 0
        self._data_n = 0

        # Sampling:
        self._sampler_idx = 0
        self._order = None

        # Data Structure:
        self._data = []
        for i in range(len(dim)):
            self._data.append(np.empty((self._max_samples, ) + dim[i]))

    def __iter__(self):
        # Shuffle data and reset counter:
        self._order = np.random.permutation(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp:self._sampler_idx]

        # Reject Batches that have less samples:
        if batch_idx.size < self._minibatch_size:
            raise StopIteration()

        out = [x[batch_idx] for x in self._data]
        return out

    def add_samples(self, data):
        assert len(data) == len(self._data)

        # Add samples:
        add_idx = self._data_idx + np.arange(data[0].shape[0])
        add_idx = np.mod(add_idx, self._max_samples)

        for i in range(len(data)):
            self._data[i][add_idx] = data[i][:]

        # Update index:
        self._data_idx = np.mod(add_idx[-1] + 1, self._max_samples)
        self._data_n = min(self._data_n + data[0].shape[0], self._max_samples)

        # Clear excessive GPU Memory:
        del data

    def shuffle(self):
        self._order = np.random.permutation(self._data_idx)
        self._sampler_idx = 0

    def get_full_mem(self):
        out = [x[:self._data_n] for x in self._data]
        return out

    def not_empty(self):
        return self._data_n > 0


class PyTorchReplayMemory(ReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim, cuda):
        super(PyTorchReplayMemory, self).__init__(max_samples, minibatch_size, dim)

        self._cuda = cuda
        for i in range(len(dim)):
            self._data[i] = torch.empty((self._max_samples,) + dim[i])

            if self._cuda:
                self._data[i] = self._data[i].cuda()

    def add_samples(self, data):

        # Cast Input Data:
        tmp_data = []

        for i, x in enumerate(data):
            if isinstance(x, np.ndarray):
                x= torch.from_numpy(x).float()

            tmp_data.append(x.type_as(self._data[i]))
            # tmp_data[i] = tmp_data[i].type_as(self._data[i])

        # Add samples to the Replay Memory:
        super(PyTorchReplayMemory, self).add_samples(tmp_data)

class PyTorchTestMemory(PyTorchReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim, cuda):
        super(PyTorchTestMemory, self).__init__(max_samples, minibatch_size, dim, cuda)

    def __iter__(self):
        # Reset counter:
        self._order = np.arange(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp:self._sampler_idx]
        out = [x[batch_idx] for x in self._data]
        return out


class RandomBuffer(ReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim_input, dim_output, enforce_max_batch_size=False):
        super(RandomBuffer, self).__init__(max_samples, minibatch_size, dim_input, dim_output)

        # Parameters:
        self._enforce_max_batch_size = enforce_max_batch_size

    def get_mini_batch(self):
        if self._data_n == 0 or (self._enforce_max_batch_size and self._data_n < self._minibatch_size):
            return None, None

        # Draw Random Mini-Batch
        idx = random.choice(self._data_n, min(self._minibatch_size, self._data_n))
        x_batch = np.array(self._x[idx], copy=True)
        y_batch = np.array(self._y[idx], copy=True)

        # Note Faster with indexing:
        # This should be faster with indexing, as less memory operation are used. However, this would significantly
        # increase implementation complexity. Therefore, this is currently not planned!

        # Remove Samples from Buffer:
        after_removal_x = np.delete(self._x, idx, 0)
        after_removal_y = np.delete(self._y, idx, 0)
        self._data_n -= idx.size

        if self._data_n > 0:
            self._x[0:self._data_n] = after_removal_x[0:self._data_n]
            self._y[0:self._data_n] = after_removal_y[0:self._data_n]

        return x_batch, y_batch

    def __next__(self):
        raise RuntimeError

    def __iter__(self):
        raise RuntimeError


class RandomReplayMemory(ReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim_input, dim_output):
        super(RandomReplayMemory, self).__init__(max_samples, minibatch_size, dim_input, dim_output)

    def add_samples(self, x, y):
        n_samples = x.shape[0]
        assert n_samples < self._max_samples

        # Add Samples in sequential order:
        add_idx = np.arange(self._data_n, min(self._data_n + n_samples, self._max_samples))

        self._x[add_idx] = x[:add_idx.size]
        self._y[add_idx] = y[:add_idx.size]

        self._data_n += add_idx.size
        assert self._data_n <= self._max_samples

        # Add samples in random order:
        random_add_idx = random.choice(self._data_n, n_samples - add_idx.size, replace=False)

        self._x[random_add_idx] = x[add_idx.size:]
        self._y[random_add_idx] = y[add_idx.size:]

    def get_mini_batch(self):
        raise RuntimeError

    def __next__(self):
        raise RuntimeError

    def __iter__(self):
        raise RuntimeError



def init_env(args):

    # Set the NumPy Formatter:
    np.set_printoptions(suppress=True, precision=2, linewidth=500,
                        formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # Read the parameters:
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    render, load_model, save_model = bool(args.r[0]), bool(args.l[0]), bool(args.m[0])

    cuda_flag = cuda_flag and torch.cuda.is_available()

    # Set the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA Device:
    if torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return seed, cuda_flag, render, load_model, save_model


def load_dataset(n_characters=3, filename = "D:/UGP-1/lutter codes/data/character_data.pickle", test_label=("e", "q", "v")):

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    

    n_dof = 2

    # Split the dataset in train and test set:

    # Random Test Set:
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # Specified Test Set:
    # test_char = ["e", "q", "v"]
    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(len(data["labels"])):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))

            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))

            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
           (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
           divider, dt_mean


def parition_params(module_name, name, value, key):
    return module_name.split("/")[0] == key

def get_params(params, key):
    return hk.data_structures.partition(jax.partial(parition_params, key=key), params)

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
}


class LowTri:

    def __init__(self, m):

        # Calculate lower triangular matrix indices using numpy
        self._m = m
        self._idx = np.tril_indices(self._m)

    def __call__(self, l):
        batch_size = l.shape[0]
        self._L = torch.zeros(batch_size, self._m, self._m).type_as(l)

        # Assign values to matrix:
        self._L[:batch_size, self._idx[0], self._idx[1]] = l[:]
        return self._L[:batch_size]


class SoftplusDer(nn.Module):
    def __init__(self, beta=1.):
        super(SoftplusDer, self).__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -20., 20.)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)

        if torch.isnan(out).any():
            print("SoftPlus Forward output is NaN.")
        return out


class ReLUDer(nn.Module):
    def __init__(self):
        super(ReLUDer, self).__init__()

    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class LinearDer(nn.Module):
    def __init__(self):
        super(LinearDer, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 1, 1)


class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class CosDer(nn.Module):
    def __init__(self):
        super(CosDer, self).__init__()

    def forward(self, x):
        return -torch.sin(x)


class LagrangianLayer(nn.Module):

    def __init__(self, input_size, n_dof, activation="ReLu"):
        super(LagrangianLayer, self).__init__()

        # Create layer weights and biases:
        self.n_dof = n_dof
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias = nn.Parameter(torch.Tensor(n_dof))

        # Initialize activation function and its derivative:
        if activation == "ReLu":
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)

        elif activation == "Cos":
            self.g = Cos()
            self.g_prime = CosDer()

        elif activation == "Linear":
            self.g = Linear()
            self.g_prime = LinearDer()

        else:
            raise ValueError("Activation Type must be in ['Linear', 'ReLu', 'SoftPlus', 'Cos'] but is {0}".format(self.activation))

    def forward(self, q, der_prev):
        # Apply Affine Transformation:
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_dof, 1) * self.weight, der_prev)
        return out, der


class DeepLagrangianNetwork(nn.Module):

    def __init__(self, n_dof, **kwargs):
        super(DeepLagrangianNetwork, self).__init__()

        # Read optional arguments:
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self._b0 = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)

        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon = kwargs.get("diagonal_epsilon", 1.e-5)

        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":

            # Construct initialization function:
            def init_hidden(layer):

                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":

            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError("Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] but is {0}".format(self._w_init))

        # Compute In- / Output Sizes:
        self.n_dof = n_dof
        self.m = int((n_dof ** 2 + n_dof) / 2)

        # Compute non-zero elements of L:
        l_output_size = int((self.n_dof ** 2 + self.n_dof) / 2)
        l_lower_size = l_output_size - self.n_dof

        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract([x not in idx_diag for x in np.arange(l_output_size)], np.arange(l_output_size))

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        # create it once and only apply repeat, this may decrease memory allocation
        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        # print(self._eye)
        self.low_tri = LowTri(self.n_dof)

        # Create Network:
        self.layers = nn.ModuleList()
        non_linearity = kwargs.get("activation", "ReLu")

        # Create Input Layer:
        self.layers.append(LagrangianLayer(self.n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(LagrangianLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        # Create output Layer:
        self.net_g = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)

        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)

        # The diagonal must be non-negative. Therefore, the non-linearity is set to ReLu.
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation="ReLu")
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

    def forward(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[6] + out[7]

        return tau_pred, dEdt

    def _dyn_model(self, q, qd, qdd):
        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)
        # print(qd_4d,qd_3d)

        # Create initial derivative of dq/dq.
        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)

        # Compute shared network between l & g:
        y, der = self.layers[0](q, der)

        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)

        # Compute the network heads including the corresponding derivative:
        l_lower, der_l_lower = self.net_lo(y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        , der)
        l_diag, der_l_diag = self.net_ld(y, der)

        # Compute the potential energy and the gravitational force:
        V, der_V = self.net_g(y, der)
        V = V.squeeze()
        g = der_V.squeeze()
        # print('V:', V, ' and g:', g, 'V shape = ', V.shape, 'g shape = ', g.shape)

        # Assemble l and der_l
        l_diag = l_diag
        l = torch.cat((l_diag, l_lower), 1)[:, self._idx]
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]

        # Compute H:
        L = self.low_tri(l)
        LT = L.transpose(dim0=1, dim1=2)
        H = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        # Calculate dH/dt
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)

        # Calculate dH/dq:
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)).reshape(-1, self.n_dof, self.n_dof, self.n_dof)
        Hdq = torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof)) + torch.matmul(L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3))

        # Compute the Coriolis & Centrifugal forces:
        Hdt_qd = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), torch.matmul(Hdq, qd_4d)).view(-1, self.n_dof)
        c = Hdt_qd - 1. / 2. * quad_dq

        # Compute the Torque using the inverse model:
        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        tau_pred = H_qdd + c + g

        # Compute kinetic energy T
        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = 1. / 2. * torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qd.view(-1, 1, self.n_dof, 1)).view(-1)

        # Compute dT/dt:
        qd_H_qdd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qdd.view(-1, 1, self.n_dof, 1)).view(-1)
        qd_Hdt_qd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), Hdt_qd.view(-1, 1, self.n_dof, 1)).view(-1)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd

        # Compute dV/dt
        dVdt = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), g.view(-1, 1, self.n_dof, 1)).view(-1)
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    def inv_dyn(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        return tau_pred

    def for_dyn(self, q, qd, tau):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        H, c, g = out[1], out[2], out[3]

        # Compute Acceleration, e.g., forward model:
        invH = torch.inverse(H)
        qdd_pred = torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        return qdd_pred

    def energy(self, q, qd):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        E = out[4] + out[5]
        return E

    def energy_dot(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        dEdt = out[6] + out[7]
        return dEdt

    def cuda(self, device=None):

        # Move the Network to the GPU:
        super(DeepLagrangianNetwork, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):

        # Move the Network to the CPU:
        super(DeepLagrangianNetwork, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self

if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[42, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[0, ], help="Save the DeLaN model")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    # Read the dataset:
    n_dof = 2
    train_data, test_data, divider,dt_mean = load_dataset()
    train_labels, train_qp, train_qv, train_qa, train_p,train_pd, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa,test_p, test_pd, test_tau, test_m, test_c, test_g = test_data
    # print('hi')
    # print(train_p)
    # print(train_pd)
    # print(train_labels)
    # plt.plot(train_qp[:,0])
    # plt.plot(train_qp[:,1])
    # plt.show()
    # print(int(test_qp.shape[0]))



    print("\n\n################################################")
    print("Characters:")
    print("   Test Characters = {0}".format(test_labels))
    print("  Train Characters = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    print("\n################################################")
    print("Training Deep Lagrangian Networks (DeLaN):")

    # Construct Hyperparameters:
    hyper = {'n_width': 64,
             'n_depth': 2,
             'diagonal_epsilon': 0.01,
             'activation': 'SoftPlus',
             'b_init': 1.e-4,
             'b_diag_init': 0.001,
             'w_init': 'xavier_normal',
             'gain_hidden': np.sqrt(2.),
             'gain_output': 0.1,
             'n_minibatch': 512,
             'learning_rate': 5.e-04,
             'weight_decay': 1.e-5,
             'max_epoch': 100}

    # Load existing model parameters:
    load_model=True
    if load_model:
        load_file = "D:/UGP-1/delan_model4_avinash_sine_10000ep_dte4_h2_w64.torch"
        state = torch.load(load_file)

        delan_model = DeepLagrangianNetwork(n_dof, **state['hyper'])
        # print(state["epoch"])
        delan_model.load_state_dict(state['state_dict'])
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    else:
        # Construct DeLaN:
        delan_model = DeepLagrangianNetwork(n_dof, **hyper)
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    # Generate & Initialize the Optimizer:
    optimizer = torch.optim.Adam(delan_model.parameters(),
                                 lr=hyper["learning_rate"],
                                 weight_decay=hyper["weight_decay"],
                                 amsgrad=True)

    # Generate Replay Memory:
    mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ))
    mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
    mem.add_samples([train_qp, train_qv, train_qa, train_tau])

    # Start Training Loop:
    t0_start = time.perf_counter()

    epoch_i = 0
    while epoch_i < hyper['max_epoch'] and not load_model:
        l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
        l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
        l_mem, n_batches = 0.0, 0.0
        # print(epoch_i)

        for q, qd, qdd, tau in mem:
            t0_batch = time.perf_counter()

            # Reset gradients:
            optimizer.zero_grad()

            # Compute the Rigid Body Dynamics Model:
            tau_hat, dEdt_hat = delan_model(q, qd, qdd)

            # Compute the loss of the Euler-Lagrange Differential Equation:
            err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
            l_mean_inv_dyn = torch.mean(err_inv)
            l_var_inv_dyn = torch.var(err_inv)
            # print(tau_hat.shape,err_inv.shape,l_mean_inv_dyn.shape)
        

            # Compute the loss of the Power Conservation:
            dEdt = torch.matmul(qd.view(-1, 2, 1).transpose(dim0=1, dim1=2), tau.view(-1, 2, 1)).view(-1)
            err_dEdt = (dEdt_hat - dEdt) ** 2
            l_mean_dEdt = torch.mean(err_dEdt)
            l_var_dEdt = torch.var(err_dEdt)

            # Compute gradients & update the weights:
            loss = l_mean_inv_dyn + l_mem_mean_dEdt
            loss.backward()
            optimizer.step()

            # Update internal data:
            n_batches += 1
            l_mem += loss.item()
            l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
            l_mem_var_inv_dyn += l_var_inv_dyn.item()
            l_mem_mean_dEdt += l_mean_dEdt.item()
            l_mem_var_dEdt += l_var_dEdt.item()

            t_batch = time.perf_counter() - t0_batch

        # Update Epoch Loss & Computation Time:
        l_mem_mean_inv_dyn /= float(n_batches)
        l_mem_var_inv_dyn /= float(n_batches)
        l_mem_mean_dEdt /= float(n_batches)
        l_mem_var_dEdt /= float(n_batches)
        l_mem /= float(n_batches)
        # print('n_batches',n_batches)
        epoch_i += 1

        if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
            print("Loss = {0:.3e}".format(l_mem), end=", ")
            print("Inv Dyn = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)), end=", ")
            print("Power Con = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))

    # Save the Model:
    # save_model=1
    if save_model:
        torch.save({"epoch": epoch_i,
                    "hyper": hyper,
                    "state_dict": delan_model.state_dict()},
                    "D:/UGP-1/delan_model.torch")

    print("\n################################################")
    print("Evaluating DeLaN:")

    # Compute the inertial, centrifugal & gravitational torque using batched samples
    t0_batch = time.perf_counter()

    # Convert NumPy samples to torch:
    q = torch.from_numpy(test_qp).float().to(delan_model.device)
    qd = torch.from_numpy(test_qv).float().to(delan_model.device)
    qdd = torch.from_numpy(test_qa).float().to(delan_model.device)
    zeros = torch.zeros_like(q).float().to(delan_model.device)

    # Compute the torque decomposition:
    with torch.no_grad():
        delan_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
        delan_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_g
        delan_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_g
        _,delan_H,_,_,_,_,_,_ = delan_model._dyn_model(q,qd,qdd)
    
    # # print(delan_H[:,0,0])
    # plt.plot(delan_H[:,0,0])
    # # plt.legend(['1st element'])
    # plt.plot(delan_H[:,0,1])
    # # plt.legend(['2nd element'])
    # plt.plot(delan_H[:,1,0])
    # # plt.legend(['3rd element'])
    # plt.plot(delan_H[:,1,1])
    # # plt.legend(['4th element'])
    # plt.plot(test_qp[:,0])
    # # plt.legend(['test q1'])
    # plt.plot(test_qp[:,1])
    # plt.legend(["1st element","2nd element","3rd element","4th element","test q1","test q2"])
    # plt.show()

    t_batch = (time.perf_counter() - t0_batch) / (3. * float(test_qp.shape[0]))

    # Move model to the CPU:
    delan_model.cpu()

    # Compute the joint torque using single samples on the CPU. The results is done using only single samples to
    # imitate the online control-loop. These online computation are performed on the CPU as this is faster for single
    # samples.

    delan_tau, delan_dEdt = np.zeros(test_qp.shape), np.zeros((test_qp.shape[0], 1))
    t0_evaluation = time.perf_counter()
    for i in range(test_qp.shape[0]):

        with torch.no_grad():

            # Convert NumPy samples to torch:
            q = torch.from_numpy(test_qp[i]).float().view(1, -1)
            qd = torch.from_numpy(test_qv[i]).float().view(1, -1)
            qdd = torch.from_numpy(test_qa[i]).float().view(1, -1)

            # Compute predicted torque:
            out = delan_model(q, qd, qdd)
            delan_tau[i] = out[0].cpu().numpy().squeeze()
            delan_dEdt[i] = out[1].cpu().numpy()

    t_eval = (time.perf_counter() - t0_evaluation) / float(test_qp.shape[0])

    # Compute Errors:
    test_dEdt = np.sum(test_tau * test_qv, axis=1).reshape((-1, 1))
    err_g = 1. / float(test_qp.shape[0]) * np.sum((delan_g - test_g) ** 2)
    err_m = 1. / float(test_qp.shape[0]) * np.sum((delan_m - test_m) ** 2)
    err_c = 1. / float(test_qp.shape[0]) * np.sum((delan_c - test_c) ** 2)
    err_tau = 1. / float(test_qp.shape[0]) * np.sum((delan_tau - test_tau) ** 2)
    err_dEdt = 1. / float(test_qp.shape[0]) * np.sum((delan_dEdt - test_dEdt) ** 2)

    print("\nPerformance:")
    print("                Torque MSE = {0:.3e}".format(err_tau))
    print("              Inertial MSE = {0:.3e}".format(err_m))
    print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
    print("         Gravitational MSE = {0:.3e}".format(err_g))
    print("    Power Conservation MSE = {0:.3e}".format(err_dEdt))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1./t_eval))

    print("\n################################################")
    print("Plotting Performance:")
    
import pybullet as bullet
import math
# Parameters:
robot_base = [1., 0., 0.]
robot_orientation = [0., 0., 0., 1.]
delta_t = 0.0001
start = 0.0
end = 10.0
steps = int((end - start) / delta_t)
t=[0.]*steps

# Initialize Bullet Simulator
id_simulator = bullet.connect(bullet.GUI)  # or bullet.DIRECT for non-graphical version
bullet.setTimeStep(delta_t)
bullet.setGravity(0,0,-10)
# planeId = bullet.loadURDF("plane.urdf")

id_revolute_joints=[0,1,2]
 
id_robot3=bullet.loadURDF("D:\\UGP-1\\svan_RR_leg_rig_urdf\\urdf\\svan_RR_leg_rig_urdf.urdf",[0.,0.,0.], robot_orientation,useFixedBase=True)
id_robot_2=bullet.loadURDF("D:\\UGP-1\\svan_RR_leg_rig_urdf\\urdf\\svan_RR_leg_rig_urdf.urdf",[1.,0.,0.], robot_orientation,useFixedBase=True)

bullet.changeDynamics(id_robot_2, 0, linearDamping=0, angularDamping=0)
bullet.changeDynamics(id_robot_2, 1, linearDamping=0, angularDamping=0)
bullet.changeDynamics(id_robot_2, 2, linearDamping=0, angularDamping=0)
bullet.setJointMotorControlArray(id_robot_2,
                                 id_revolute_joints,
                                 bullet.VELOCITY_CONTROL,
                                 forces=[0.0, 0.0, 0.0])

q_pos_test = [[0.] * steps, [0.] * steps, [0.] * steps]
q_vel_test = [[0.] * steps, [0.] * steps, [0.] * steps]
q_tor_test = [[0.] * steps, [0.] * steps, [0.] * steps]

# Target Positions:
q_pos_desired_test = [[0.] * steps, [0.] * steps]
q_vel_desired_test = [[0.] * steps, [0.] * steps]
q_acc_desired_test = [[0.] * steps, [0.] * steps, [0.] * steps]
A=1
A1=0.56*A
A2=0.8*A
offset_1=0.7
offset_2=-1.2
freq=1
k=1/freq
# A=0.7
# A1=A/(2*math.pi)
# A2=A/(2*math.pi)
# offset_1=0
# offset_2=-1
# freq=0.2
# k=1/freq

for s in range(steps):
  t[s] = start + s * delta_t
  q_pos_desired_test[0][s] = A1* math.sin(2. * math.pi * t[s]/k ) + offset_1
  q_pos_desired_test[1][s] = -A2 * (math.sin(2. * math.pi * t[s]/k)) + offset_2

  q_vel_desired_test[0][s] = A1*2.*math.pi*(math.cos(2. * math.pi * t[s]/k))/k
  q_vel_desired_test[1][s] = -A2*2.*math.pi*math.cos(2. * math.pi * t[s]/k)/k
  
  q_acc_desired_test[0][s]= 0
  q_acc_desired_test[1][s] = -A1*(2. * math.pi)*(2. * math.pi) * math.sin(2. * math.pi * t[s]/k)/(k*k)
  q_acc_desired_test[2][s] = A2*(2. * math.pi) *(2.*math.pi) * math.sin(2. * math.pi * t[s]/k)/(k*k)
# Do Torque Control:
# log_id = bullet.startStateLogging(bullet.STATE_LOGGING_VIDEO_MP4, "D:\\UGP-1\\simulation_ID_function.mp4")
# A = 1.1
# freq=0.1
# k=1/freq
# for s in range(steps):
#   t[s] = start + s * delta_t
#   q_pos_desired_test[0][s] = A * 1. / (2. * math.pi) * math.sin(2. * math.pi * t[s]/k )      
#   q_pos_desired_test[1][s] = A * (-1.) / (2. * math.pi) * (math.cos(2. * math.pi * t[s]/k) - 1.0)

#   q_vel_desired_test[0][s] = A * (math.cos(2. * math.pi * t[s]/k))/k #- 1.
#   q_vel_desired_test[1][s] = A * math.sin(2. * math.pi * t[s]/k)/k
  
#   q_acc_desired_test[0][s]= 0
#   q_acc_desired_test[1][s] = -2. * A * math.pi * math.sin(2. * math.pi * t[s]/k)/(k*k)
#   q_acc_desired_test[2][s] = 2. * A * math.pi * math.cos(2. * math.pi * t[s]/k)/(k*k)

# bullet.setRealTimeSimulation(1)
# bullet.enableJointForceTorqueSensor(id_robot_2,0,1)
# bullet.enableJointForceTorqueSensor(id_robot_2,1,1)
# bullet.enableJointForceTorqueSensor(id_robot_2,2,1)
# q_t_test=[[0.]*steps,[0.]*steps,[0.]*steps]
for i in range(len(t)):
  if i==0:
    bullet.resetJointState(id_robot_2, 1,q_pos_desired_test[0][i],q_vel_desired_test[0][i])
    bullet.resetJointState(id_robot_2, 2,q_pos_desired_test[1][i],q_vel_desired_test[1][i])

  # Read Sensor States:
  joint_states = bullet.getJointStates(id_robot_2, id_revolute_joints)
#   print(joint_states[1][2])
  

  q_pos_test[0][i] = joint_states[0][0]
  q_pos_test[1][i] = joint_states[1][0]
  q_pos_test[2][i] = joint_states[2][0]

  q_vel_test[0][i] = joint_states[0][1]
  q_vel_test[1][i] = joint_states[1][1]
  q_vel_test[2][i] = joint_states[2][1]

#   q_t_test[0][i]=joint_states[0][3]
#   q_t_test[1][i]=joint_states[1][2][5]
#   q_t_test[2][i]=joint_states[2][2][4]
  
  # Computing the torque from inverse dynamics:
  obj_pos_test = [q_pos_test[0][i], q_pos_test[1][i], q_pos_test[2][i]]
  obj_vel_test = [q_vel_test[0][i], q_vel_test[1][i], q_vel_test[2][i]]
#   obj_pos_test = [q_pos_test[0][i], q_pos_desired_test[0][i], q_pos_desired_test[1][i]]
#   obj_vel_test = [q_vel_test[0][i], q_vel_desired_test[0][i], q_vel_desired_test[1][i]]
  obj_acc_test = [q_acc_desired_test[0][i], q_acc_desired_test[1][i], q_acc_desired_test[2][i]]

  torque_test = bullet.calculateInverseDynamics(id_robot_2, obj_pos_test, obj_vel_test, obj_acc_test)

  q_tor_test[0][i] = torque_test[0]
  q_tor_test[1][i] = torque_test[1] 
  q_tor_test[2][i] = torque_test[2] 
  
  # Set the Joint Torques:
  bullet.setJointMotorControlArray(id_robot_2,
                                   id_revolute_joints,
                                   bullet.TORQUE_CONTROL,
                                   forces=[torque_test[0], torque_test[1], torque_test[2]])
#   bullet.setJointMotorControlArray(id_robot,
#                                    [0,1,2],
#                                    bullet.TORQUE_CONTROL,
#                                    forces=[0, torque_test[1], torque_test[2]])

  # Step Simulation
  bullet.stepSimulation()
#   time.sleep(1/240)
# print('hi')
if 0:
  figure = plt.figure(figsize=[15, 4.5])
  figure.subplots_adjust(left=0.05, bottom=0.11, right=0.87, top=0.9, wspace=0.4, hspace=0.55)

  ax_pos = figure.add_subplot(141)
  ax_pos.set_title("Joint Position")
  ax_pos.plot(t, q_pos_desired_test[0], '--r', lw=2, label='Desired q1')
  ax_pos.plot(t, q_pos_desired_test[1], '--b', lw=2, label='Desired q2')   
  ax_pos.plot(t, q_pos_test[1], '-r', lw=1, label='Measured q1')
  ax_pos.plot(t, q_pos_test[2], '-b', lw=1, label='Measured q2')
#   ax_pos.set_ylim(-1., 1.)
  ax_pos.legend()

  ax_vel = figure.add_subplot(142)
  ax_vel.set_title("Joint Velocity")
  ax_vel.plot(t, q_vel_desired_test[0], '--r', lw=2, label='Desired q1')
  ax_vel.plot(t, q_vel_desired_test[1], '--b', lw=2, label='Desired q2')
  ax_vel.plot(t, q_vel_test[1], '-r', lw=1, label='Measured q1')
  ax_vel.plot(t, q_vel_test[2], '-b', lw=1, label='Measured q2')
#   ax_vel.set_ylim(-2., 2.)
  ax_vel.legend()

  ax_acc = figure.add_subplot(143)
  ax_acc.set_title("Joint Acceleration")
  ax_acc.plot(t, q_acc_desired_test[1], '--r', lw=4, label='Desired q1')
  ax_acc.plot(t, q_acc_desired_test[2], '--b', lw=4, label='Desired q2')
#   ax_acc.set_ylim(-10., 10.)
  ax_acc.legend()

  ax_tor = figure.add_subplot(144)
  ax_tor.set_title("Executed Torque")
#   ax_tor.plot(t, q_tor[0], '-g', lw=2, label='Torque q0')
  ax_tor.plot(t, q_tor_test[1], '-r', lw=2, label='Delan Torque q1')
  ax_tor.plot(t, q_tor_test[2], '-b', lw=2, label='Delan Torque q2')
#   ax_tor.plot(t, q_t_test[1], '--r', lw=2, label='Sensed Torque q1')
#   ax_tor.plot(t, q_t_test[2], '--b', lw=2, label='Sensed Torque q2')
  ax_tor.legend()


  plt.pause(1000)

# print('hi1')
q_pos_array = np.array([q_pos_test[1],q_pos_test[2]])
q_pos_desired_test_array = np.array(q_pos_desired_test)

mse_error = np.mean((q_pos_array - q_pos_desired_test_array)**2)
mae_error = np.mean(np.abs(q_pos_array - q_pos_desired_test_array))
print('Mean Squared Error: ',mse_error,'                 Mean Absolute Error:',mae_error,'         (These are for joint velocity for A=', A, ' and frequency=', freq, 'Hz )')

bullet.changeDynamics(id_robot3, 0, linearDamping=0, angularDamping=0)
bullet.changeDynamics(id_robot3, 1, linearDamping=0, angularDamping=0)
bullet.changeDynamics(id_robot3, 2, linearDamping=0, angularDamping=0)

jointTypeNames = [
    "JOINT_REVOLUTE", "JOINT_PRISMATIC", "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED",
    "JOINT_POINT2POINT", "JOINT_GEAR"
]

# Disable the motors for torque control:
bullet.setJointMotorControlArray(id_robot3,
                                 id_revolute_joints,
                                 bullet.VELOCITY_CONTROL,
                                 forces=[0.0, 0.0, 0.0])

q_pos = [[0.] * steps, [0.] * steps, [0.] * steps]
q_vel = [[0.] * steps, [0.] * steps, [0.] * steps]
q_tor = [[0.] * steps, [0.] * steps, [0.] * steps]
delan_tau= np.zeros([steps,2])

# Target Positions:
# q_pos_desired_test = [[0.] * steps, [0.] * steps]
# q_vel_desired_test = [[0.] * steps, [0.] * steps]
# q_acc_desired_test = [[0.] * steps, [0.] * steps, [0.] * steps]
# k=10
    # Convert NumPy samples to torch:
q = torch.from_numpy(np.transpose(np.array([q_pos_desired_test[0],q_pos_desired_test[1]]))).float().to(delan_model.device)
qd = torch.from_numpy(np.transpose(np.array([q_vel_desired_test[0],q_vel_desired_test[1]]))).float().to(delan_model.device)
qdd = torch.from_numpy(np.transpose(np.array([q_acc_desired_test[1],q_acc_desired_test[2]]))).float().to(delan_model.device)
zeros = torch.zeros_like(q).float().to(delan_model.device)

    # Compute the torque decomposition:
with torch.no_grad():
    # delan_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
    # delan_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_g
    # delan_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_g
    _,delan_H,_,_,_,_,_,_ = delan_model._dyn_model(q,zeros,zeros)

    # # print(delan_H[:,0,0])
    # plt.plot(delan_H[:,0,0])
    # # plt.legend(['1st element'])
    # plt.plot(delan_H[:,0,1])
    # # plt.legend(['2nd element'])
    # plt.plot(delan_H[:,1,0])
    # # plt.legend(['3rd element'])
    # plt.plot(delan_H[:,1,1])
    # # plt.legend(['4th element'])
    # plt.plot(test_qp[:,0])
    # # plt.legend(['test q1'])
    # plt.plot(test_qp[:,1])
    # plt.legend(["1st element","2nd element","3rd element","4th element","test q1","test q2"])
    # plt.show()
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# # Titles for each subplot
# titles = [
#     "1,1", "1,2", "1,3",
#     "2,1", "2,2", "2,3",
#     "3,1", "3,2", "3,3"
#     ]

# # Plot each element in its respective subplot
# for i in range(3):
#     for j in range(3):
#         ax = axes[i, j]
#         # ax.plot([pybullet_H[step][i][j] for step in range(steps)], label='PyBullet H')
#         if i > 0 and j > 0:
#             ax.plot([delan_H[step][i-1][j-1] for step in range(steps)], label='DeLaN H')
#         ax.set_title(titles[i * 3 + j])
#         ax.set_xlabel('Time step')
#         ax.set_ylabel('Value')
        # ax.legend()

# Adjust layout for better spacing
# plt.tight_layout()
# plt.show()

for i in range(steps):
    with torch.no_grad():

            # Convert NumPy samples to torch:
        q = torch.from_numpy(np.transpose(np.array([q_pos_desired_test[0][i],q_pos_desired_test[1][i]]))).float().view(1, -1)
        qd = torch.from_numpy(np.transpose(np.array([q_vel_desired_test[0][i],q_vel_desired_test[1][i]]))).float().view(1, -1)
        qdd = torch.from_numpy(np.transpose(np.array([q_acc_desired_test[1][i],q_acc_desired_test[2][i]]))).float().view(1, -1)

            # Compute predicted torque:
        out = delan_model(q, qd, qdd)
        delan_tau[i] = out[0].cpu().numpy().squeeze()
# bullet.setRealTimeSimulation(1)
# bullet.enableJointForceTorqueSensor(id_robot3,1,1)
# bullet.enableJointForceTorqueSensor(id_robot3,2,1)
# q_t=[[0.]*steps,[0.]*steps,[0.]*steps]
# Do Torque Control:
PD_t= [[0]*steps,[0]*steps]
for i in range(len(t)):
  if i==0:
    bullet.resetJointState(id_robot3, 1,q_pos_desired_test[0][i],q_vel_desired_test[0][i])
    bullet.resetJointState(id_robot3, 2,q_pos_desired_test[1][i],q_vel_desired_test[1][i])

  # Read Sensor States:
  joint_states = bullet.getJointStates(id_robot3, id_revolute_joints)
#   print(joint_states[1][2])
  q_pos[0][i] = joint_states[0][0]
  q_pos[1][i] = joint_states[1][0]
  q_pos[2][i] = joint_states[2][0]

  q_vel[0][i] = joint_states[0][1]
  q_vel[1][i] = joint_states[1][1]
  q_vel[2][i] = joint_states[2][1]

  kp=[0.01,0.01]
  kd=[0.005,0.005]
  PD_t[0] = kp[0]*(q_pos[1][i]-q_pos_desired_test[0][i] + kd[0]*(q_vel[1][i]-q_vel_desired_test[0][i]))
  PD_t[1] = kp[1]*(q_pos[2][i]-q_pos_desired_test[1][i] + kd[1]*(q_vel[2][i]-q_vel_desired_test[1][i]))
  
#   q_t[0][i]=joint_states[0][3]
#   q_t[1][i]=joint_states[1][3]
#   q_t[2][i]=joint_states[2][3]
#   print(q_t[2][i])
#   torque = [q_tor_test[0][i], 11*delan_tau[i][0]/14, delan_tau[i][1]]
  torque = [delan_tau[i,0], delan_tau[i,1]]
#   torque = [delan_tau[i,0]+PD_t[0], delan_tau[i,1]+PD_t[1]]

  q_tor[0][i] = torque[0]
  q_tor[1][i] = torque[1] 
#   q_tor[2][i] = torque[2] 
    # Set the Joint Torques:
  bullet.setJointMotorControlArray(id_robot3,
                                   [1,2],
                                   bullet.TORQUE_CONTROL,
                                   forces=[torque[0], torque[1]])

  # Step Simulation
  bullet.stepSimulation()
#   time.sleep(1/240)

q_pos_array = np.array([q_vel[1],q_vel[2]])
q_pos_desired_test_array = np.array(q_vel_desired_test)

mse_error = np.mean((q_pos_array - q_pos_desired_test_array)**2)
mae_error = np.mean(np.abs(q_pos_array - q_pos_desired_test_array))
print('Mean Squared Error: ',mse_error,'                 Mean Absolute Error:',mae_error,'         (These are for joint velocity for A=', A, ' and frequency=', freq, 'Hz )')
if 1:
  figure = plt.figure(figsize=[15, 4.5])
  figure.subplots_adjust(left=0.05, bottom=0.11, right=0.87, top=0.9, wspace=0.4, hspace=0.55)

  ax_pos = figure.add_subplot(141)
  ax_pos.set_title("Joint Position")
  ax_pos.plot(t, q_pos_desired_test[0], '--r', lw=2, label='Desired q1')
  ax_pos.plot(t, q_pos_desired_test[1], '--b', lw=2, label='Desired q2')   
  ax_pos.plot(t, q_pos[1], '-r', lw=1, label='Measured q1')
  ax_pos.plot(t, q_pos[2], '-b', lw=1, label='Measured q2')
#   ax_pos.set_ylim(-1., 1.)
  ax_pos.legend()

  ax_vel = figure.add_subplot(142)
  ax_vel.set_title("Joint Velocity")
  ax_vel.plot(t, q_vel_desired_test[0], '--r', lw=2, label='Desired q1')
  ax_vel.plot(t, q_vel_desired_test[1], '--b', lw=2, label='Desired q2')
  ax_vel.plot(t, q_vel[1], 'r', lw=0.1, label='Measured q1')
  ax_vel.plot(t, q_vel[2], 'b', lw=0.1, label='Measured q2')
#   ax_vel.set_ylim(-2., 2.)
  ax_vel.legend()

  ax_acc = figure.add_subplot(143)
  ax_acc.set_title("Joint Acceleration")
  ax_acc.plot(t, q_acc_desired_test[1], '--r', lw=4, label='Desired q1')
  ax_acc.plot(t, q_acc_desired_test[2], '--b', lw=4, label='Desired q2')
#   ax_acc.set_ylim(-10., 10.)
  ax_acc.legend()

  ax_tor = figure.add_subplot(144)
  ax_tor.set_title("Executed Torque")
#   ax_tor.plot(t, q_tor[0], '-g', lw=2, label='Torque q0')
  ax_tor.plot(t, q_tor[0], '-r', lw=2, label='Delan Torque q1')
  ax_tor.plot(t, q_tor[1], '-b', lw=2, label='Delan Torque q2')
#   ax_tor.plot(t, q_t[1], '--r', lw=2, label='Sensed Torque q1')
#   ax_tor.plot(t, q_t[2], '--b', lw=2, label='Sensed Torque q2')
  ax_tor.plot(t, q_tor_test[1], '--r', lw=1, label='Actual Torque q1')
  ax_tor.plot(t, q_tor_test[2], '--b', lw=1, label='Actual Torque q2')
#   ax_tor.set_ylim(-20., 20.)
  ax_tor.legend()


  plt.pause(1000)
