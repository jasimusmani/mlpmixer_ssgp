# Adapted from Motion Mixer : https://github.com/MotionMLP/MotionMixer/blob/main/h36m/mlp_mixer.py

import torch.nn as nn
import torch
import torch.nn.functional as F


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=4):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch_size = 1
        self.H = 15
        self.W = 609
    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        # H, W = input_tensor.size()
        input_tensor = input_tensor.reshape(1,self.H, self.W)
        print('input tensor: ', input_tensor.shape)
        # batch_size, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.reshape(self.batch_size, self.W, -1).mean(dim=2)
#         print('squeeze tensor: ', squeeze_tensor.shape)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
#         print('fc out 2', fc_out_2.shape)
        a, b = squeeze_tensor.size()
#         print(fc_out_2.view(a, b, 1, 1).shape)
#         output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.H = 15
        self.W = 1638

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        # H, W = input_tensor.size()
        input_tensor = torch.transpose(input_tensor,1,0)
#         print(input_tensor.shape)
        input_tensor = input_tensor.reshape(1,self.W,self.H,1)


        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, self.W, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
#         print('shape of squeeze tensor: ', squeeze_tensor.shape)

        # spatial excitation
#         print(input_tensor.size(), squeeze_tensor.size())
#         print(squeeze_tensor)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        output_tensor = output_tensor.reshape(1,self.H,self.W)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=4):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor



class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        s, h = x.shape
        x = x.reshape(1, s, h)
        bs, s, h = x.shape
        # print('input into se block: ', x.shape)
        y = self.squeeze(x).view(bs, s)
        # print('input into squeeze: ', y.shape)
        y = self.excitation(y).view(bs, s, 1)
        # print('input into excitation: ', y.shape)
        y = x * y.expand_as(x)
        y = y.reshape(s,h)
        # print('output excitation: ', y.shape)
        return y


def mish(x):
    return (x * torch.tanh(F.softplus(x)))


class MlpBlock(nn.Module):
    def __init__(self, mlp_hidden_dim, mlp_input_dim, mlp_bn_dim, activation='gelu', regularization=0,
                 initialization='none'):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_bn_dim = mlp_bn_dim
        # self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_input_dim)
        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg1 = nn.BatchNorm1d(self.mlp_bn_dim)
            self.reg2 = nn.BatchNorm1d(self.mlp_bn_dim)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == 'gelu':
            self.act1 = nn.GELU()
        elif activation == 'mish':
            self.act1 = mish  # nn.Mish()
        else:
            raise ValueError('Unknown activation function type: %s' % activation)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.fc2(x)
        if self.reg2 is not None:
            x = self.reg2(x)

        return x


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=False, use_chse=False, use_ch=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim,
                                               activation=activation, regularization=regularization,
                                               initialization=initialization)
        self.mlp_block_channel_mixing = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len,
                                                 activation=activation, regularization=regularization,
                                                 initialization=initialization)
        self.use_se = use_se
        self.use_chse = use_chse
        self.use_ch=use_ch
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)
        if self.use_chse:
            self.chse = ChannelSpatialSELayer(self.hidden_dim)
        if self.use_ch:
            self.ch = SpatialSELayer(self.hidden_dim)

        self.LN1 = nn.LayerNorm(self.hidden_dim)
        self.LN2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels
        # x = x.reshape(7, 546)
        x = x.reshape(self.seq_len, self.hidden_dim)
        y = self.LN1(x)

        y = y.transpose(0, 1)
        # y=y.reshape(525,7)
        y = self.mlp_block_token_mixing(y)

        y = y.transpose(0, 1)

        if self.use_se:
            y = self.se(y)
        if self.use_chse:
            y = self.chse(y)
        if self.use_ch:
            y = self.ch(y)

        x = x + y
        y = self.LN2(x)
        y = self.mlp_block_channel_mixing(y)

        if self.use_se:
            y = self.se(y)
        if self.use_chse:
            y = self.chse(y)
        if self.use_ch:
            # print('y shape', y.shape)
            y = self.ch(y)

        return x + y


class MixerBlock_Channel(nn.Module):
    def __init__(self, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=False, use_chse=False, use_ch=True):
        super().__init__()
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_channel_mixing = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len,
                                                 activation=activation, regularization=regularization,
                                                 initialization=initialization)
        self.use_se = use_se
        self.use_chse=use_chse
        self.use_ch=use_ch
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)
        if self.use_chse:
            self.chse = ChannelSpatialSELayer(self.hidden_dim)
        if self.use_ch:
            self.chse = SpatialSELayer(self.hidden_dim)
        self.LN2 = nn.LayerNorm(self.hidden_dim)

        # self.act1 = nn.GELU()

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels]
        y = x

        if self.use_se:
            y = self.se(y)
        if self.use_chse:
            y = self.chse(y)
        x = x + y
        y = self.LN2(x)
        y = self.mlp_block_channel_mixing(y)
        if self.use_se:
            y = self.se(y)
        if self.use_ch:
            y =self.ch(y)
        if self.use_chse:
            y =self.chse(y)

        return x + y


class MixerBlock_Token(nn.Module):
    def __init__(self, tokens_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=False, use_chse=False, use_ch=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim,
                                               activation=activation, regularization=regularization,
                                               initialization=initialization)

        self.use_se = use_se
        self.use_chse=use_chse
        self.use_ch=use_ch
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)
        if self.use_chse:
            self.use_chse=ChannelSpatialSELayer(self.hidden_dim)
        if self.use_ch:
            self.use_ch=SpatialSELayer(self.hidden_dim)

        self.LN1 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels]
        y = self.LN1(x)
        y = y.transpose(1, 2)
        y = self.mlp_block_token_mixing(y)
        y = y.transpose(1, 2)
        print('output before se: ', y.shape)
        if self.use_se:
            y = self.se(y)
        if self.use_chse:
            y = self.chse(y)
        if self.use_ch:
            y = self.ch(y)
        print('output after se: ', y.shape)
        x = x + y

        return x + y


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, hidden_dim, tokens_mlp_dim,
                 channels_mlp_dim, output, seq_len, pred_len, activation='gelu',
                 mlp_block_type='normal', regularization=0, input_size=78,
                 initialization='none', r_se=4, use_max_pooling=False,
                 use_se=False, use_chse=False, use_ch=True):

        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.input_size = input_size  # varyies with the number of joints
        # self.conv = nn.Conv1d(1, self.hidden_dim, (1, self.input_size), stride=1)
        self.conv = nn.Conv1d(1, self.hidden_dim, self.input_size, stride=1)
        self.activation = activation
        self.output = output
        self.channel_only = False  # False #True
        self.token_only = False  # False #True

        if self.channel_only:
            self.Mixer_Block = nn.ModuleList(MixerBlock_Channel(self.channels_mlp_dim, self.seq_len, self.hidden_dim,
                                                                activation=self.activation,
                                                                regularization=regularization,
                                                                initialization=initialization,
                                                                r_se=r_se, use_max_pooling=use_max_pooling,
                                                                use_se=use_se, use_ch=use_ch,use_chse=use_chse)
                                             for _ in range(num_blocks))

        if self.token_only:

            self.Mixer_Block = nn.ModuleList(MixerBlock_Token(self.tokens_mlp_dim, self.seq_len, self.hidden_dim,
                                                              activation=self.activation, regularization=regularization,
                                                              initialization=initialization,
                                                              r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se,
                                                              use_chse=use_chse, use_ch=use_ch)
                                             for _ in range(num_blocks))

        else:

            self.Mixer_Block = nn.ModuleList(MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim,
                                                        self.seq_len, self.hidden_dim, activation=self.activation,
                                                        regularization=regularization, initialization=initialization,
                                                        r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se,
                                                        use_chse=use_chse, use_ch=use_ch)
                                             for _ in range(num_blocks))

        self.LN = nn.LayerNorm(self.hidden_dim)

        # self.fc_out = nn.Linear(self.hidden_dim, self.num_classes)
        self.fc_out = nn.Linear(self.hidden_dim, self.output)

        self.pred_len = pred_len
        self.conv_out = nn.Conv1d(self.seq_len, self.pred_len, 1, stride=1)
        # self.conv_out = nn.Conv1d(7, 1, 1, stride=1)

    def forward(self, x):
        # x = x.reshape(7, 78)
        x = x.reshape(self.seq_len, self.num_classes)
        x = x.unsqueeze(1)

        y = self.conv(x)
        # print('value before mixer block', y.shape)
        y = y.squeeze(dim=1).transpose(1, 2)

        # [256, 8, 512] [bs, patches/time_steps, channels]
        for mb in self.Mixer_Block:
            y = mb(y)
        y = self.LN(y)
        # print('shape before conv out', y.shape)
        a = self.conv_out(y)
        # a = a.reshape(1, 546)
        a = a.reshape(1, self.hidden_dim)
        # print('shape conv out',a.shape)
        out = self.fc_out(a)
        # print('final shape',out.shape)

        return out

# things to change
# num_classes argument
# hidden_dim = 525
# seq_len = 8
# tokens_mlp_dim
# channels_mlp_dim = ?? need to figure it out
#