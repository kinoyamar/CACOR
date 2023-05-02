import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn.utils.rnn import pad_packed_sequence
from clutils.globals import OUTPUT_TYPE, choose_output

class MaskedLSTM(nn.Module):
    """
    LSTM model whose weights can be masked.

    Layers are described by the following names:
    'rnn' -> recurrent module;
    'out' -> linear readout layer.

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """

    def __init__(self, input_size, hidden_size, output_size, device,
                 num_layers=1, truncated_time=0, orthogonal=False):
        """
        Optional Parameters
        ----------
        num_layers : int, optional
            Number of LSTM layers, by default 1
        truncated_time : int, optional
            Time step to backpropagate. Backpropagate for full sequence when <= 0, 
            by default 0
        orthogonal : bool, optional
            If True, uses orthogonal initialization for LSTM hidden weights, by default False
        """

        super(MaskedLSTM, self).__init__()

        self.output_type = OUTPUT_TYPE.LAST_OUT
        self.is_recurrent = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.orthogonal = orthogonal
        self.truncated_time = truncated_time

        self.layers = nn.ModuleDict([])

        self.layers.update([ 
            ['rnn', nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True) ]
        ])
        if self.orthogonal:
            for _, hh, _, _ in self.layers['rnn'].all_weights:
                # lstm divides hidden matrix into 4 chunks
                # https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
                for j in range(0, hh.size(0), self.hidden_size): 
                    nn.init.orthogonal_(hh[j:j+self.hidden_size])

        self.layers.update([
            ['out', nn.Linear(self.hidden_size, self.output_size) ]
        ])

        self.layers = self.layers.to(self.device)

        # Initialize masks.
        mask_ih = torch.ones((self.hidden_size*4, self.input_size), dtype=int, device=device)
        mask_hh = torch.ones((self.hidden_size*4, self.hidden_size), dtype=int, device=device)
        mask_out = torch.ones((self.output_size, self.hidden_size), dtype=int, device=device)
        for i in range(self.num_layers):
            prune.custom_from_mask(self.layers['rnn'], f'weight_ih_l{i}', mask_ih if i==0 else mask_hh)
            prune.custom_from_mask(self.layers['rnn'], f'weight_hh_l{i}', mask_hh)
        prune.custom_from_mask(self.layers['out'], 'weight', mask_out)

    def forward(self, x, h=None, truncated_time=None):
        """
        Parameters
        ----------
        x : Tensor
            (batch_size, seq_len, input_size)
        h : Tensor, optional
            hidden state of the recurrent module, by default None
        truncated_time : int, optional
            Time step to backpropagate, by default None

        Returns
        -------
        out : Tensor
            (batch_size, output_size)
        """
        
        tr_time = truncated_time if truncated_time else self.truncated_time

        if tr_time > 0:
            with torch.no_grad():
                if h:
                    out_h1, h1 = self.layers['rnn'](x[:, :-tr_time, :], h)
                else:
                    out_h1, h1 = self.layers['rnn'](x[:, :-tr_time, :])

            out_h2, h2 = self.layers['rnn'](x[:, -tr_time:, :], h1)
            out = self.layers['out'](out_h2)
            out_h = torch.cat((out_h1, out_h2), dim=0)

        else:
            if h:
                out_h, h = self.layers['rnn'](x, h)
            else:
                out_h, h = self.layers['rnn'](x)

            out = self.layers['out'](out_h)

        return choose_output(out, out_h, self.output_type)

    def reset_memory_state(self, batch_size):
        """
        Parameters
        ----------
        batch_size : int
            Size of current batch.

        Returns
        -------
        hidden_states : (Tensor, Tensor)
            Hidden states of the recurrent module (h, c), reset to zeros.
        """

        h = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        )

        return h

    def get_layers(self):
        return self.layers.values()

    def set_masks(self, masks):
        """
        Apply masking to LSTM weights.

        Parameters
        ----------
        masks : list(Tensor)
            List of masking tensors. 
            (i*2)th element is mask for 'weight_ih_l{i}'. 
            (i*2+1)th element is mask for 'weight_hh_l{i}'. 
            If there is an odd number of elements, the last element is mask for linear output layer.
        """

        for i in range(len(masks) // 2):
            masks[i*2] = masks[i*2].to(self.device)
            masks[i*2+1] = masks[i*2+1].to(self.device)
            setattr(self.layers['rnn'], f'weight_ih_l{i}_mask', masks[i*2])
            setattr(self.layers['rnn'], f'weight_hh_l{i}_mask', masks[i*2+1])

        if len(masks) % 2 == 1:
            masks[-1] = masks[-1].to(self.device)
            setattr(self.layers['out'], 'weight_mask', masks[-1])

    @torch.no_grad()
    def get_weight_ratios(self):
        """
        Get a list of weight ratios.

        Returns
        -------
        list(float)
            A list of weight ratios.
            (i*2)th element is of 'weight_ih_l{i}' from 'rnn' module. 
            (i*2+1)th element is of 'weight_hh_l{i}'from 'rnn' module. 
            (-2)th element is of 'weight' from 'out' module.
            (-1)th element is overall weight ratio (bias NOT included).
        """

        weight_ratios = []
        num_all_parameters = 0
        num_parameters = 0

        # RNN layers
        for layer in range(self.num_layers):
            i_size = self.input_size if layer == 0 else self.hidden_size
            h_size = self.hidden_size

            num_all_parameters_ih = 4 * h_size * i_size
            num_all_parameters_hh = 4 * h_size * h_size

            mask_ih = getattr(self.layers['rnn'], f'weight_ih_l{layer}_mask')
            mask_hh = getattr(self.layers['rnn'], f'weight_hh_l{layer}_mask')

            num_parameters_ih = mask_ih.sum().item()
            num_parameters_hh = mask_hh.sum().item()

            weight_ratio_ih = num_parameters_ih / num_all_parameters_ih
            weight_ratio_hh = num_parameters_hh / num_all_parameters_hh

            weight_ratios.append(weight_ratio_ih)
            weight_ratios.append(weight_ratio_hh)

            num_all_parameters += num_all_parameters_ih + num_all_parameters_hh
            num_parameters += num_parameters_ih + num_parameters_hh

        # Output layer
        num_all_parameters_out = self.hidden_size * self.output_size
        mask_out = getattr(self.layers['out'], 'weight_mask')
        num_parameters_out = mask_out.sum().item()
        weight_ratio_out = num_parameters_out / num_all_parameters_out
        weight_ratios.append(weight_ratio_out)
        num_all_parameters += num_all_parameters_out
        num_parameters += num_parameters_out

        # All
        overall_weight_ratio = num_parameters / num_all_parameters
        weight_ratios.append(overall_weight_ratio)

        return weight_ratios

class MaskedSketchLSTM(MaskedLSTM):
    """
    MaskedLSTM that deals with PackedSequences
    """
    
    def forward(self, x, h=None):
        '''
        :param x: PackedSequence
        :param h: hidden state of the recurrent module

        :return out: (batch_size, seq_len, directions*hidden_size)
        :return h: hidden state of the recurrent module
        '''

        x, lengths = pad_packed_sequence(x, batch_first=True)
        lengths = lengths.to(x.device)

        if h:
            out_h, h = self.layers['rnn'](x, h)
        else:
            out_h, h = self.layers['rnn'](x)

        # out_h, lengths = pad_packed_sequence(out_h, batch_first=True)
        # lengths = lengths.to(out_h.device)
        out_h = torch.gather(out_h, 1, (lengths-1).unsqueeze(1).unsqueeze(1).expand(out_h.size(0), 1, out_h.size(2))).squeeze(1)

        out = self.layers['out'](out_h)

        return out


def create_lstm_masks(masks):
    tensor_masks = []
    for i in range(1, len(masks) - 1):
        tensor_masks.append(torch.from_numpy(np.tile(masks[i].astype(np.float32).T, (4, 1))).clone())
    tensor_masks.append(torch.from_numpy(masks[-1].astype(np.float32).T).clone())
    return tensor_masks

class MaskedLSTMWeightRatioLogger:
    def __init__(self, num_rnn_layers, file_dir='weight_ratios', file_name='training.csv', init_file=True, start_step=0):
        self.num_rnn_layers = num_rnn_layers
        self.file_dir = file_dir
        self.file_name = file_name
        self.step = start_step

        if init_file:
            os.makedirs(self.file_dir, exist_ok=True)
            with open(os.path.join(self.file_dir, self.file_name), 'w') as f:
                print('step', file=f, end='')
                for i in range(self.num_rnn_layers):
                    print(f',ih_{i},hh_{i}', file=f, end='')
                print(',out,all', file=f)
    
    def log(self, weight_ratios, step=None):
        with open(os.path.join(self.file_dir, self.file_name), 'a') as f:
            print(f'{step if step else self.step}', end='', file=f)
            for value in weight_ratios:
                print(f',{value}', end='', file=f)
            print(file=f)
        self.step += 1
