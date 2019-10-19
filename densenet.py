import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_c, k, drop_rate=0.2):
        super(DenseLayer, self).__init__()
        magic = 4 # to create 4k channels internally
        self.add_module('norm1', nn.BatchNorm2d(in_c))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv1', nn.Conv2d(in_c, magic * k,
            kernel_size=1, stride=1, bias=False))
        # after conv1 it outputs the same size

        self.add_module('norm2', nn.BatchNorm2d(magic * k))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(magic * k, k,
            kernel_size=3, stride=1, padding=1, bias=False))
        # after conv2 it outputs the same size too
        # (out_w = (in_w + 1 + 1 - 3)/1 + 1 = in_w)

        self.drop_rate = drop_rate

    def forward(self, *prev):
        concated = torch.cat(prev, 1) # concate at the channel dimension

        # the DenseLayer does not change output size (see above comments)
        # it does change the channel from in_c -> 4k -> k.
        bottleneck = self.conv1(self.relu1(self.norm1(concated)))
        out = self.conv2(self.relu2(self.norm2(bottleneck)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_c, growth_rate):
        super(DenseBlock, self).__init__()
        for l in range(n_layers):
            layer = DenseLayer(
                in_c + l * growth_rate,
                k = growth_rate
            )
            self.add_module('dense-layer-%d' % (l + 1), layer)

    def forward(self, inputs):
        prev_inputs = [inputs]
        for name, layer in self.named_children():
            # pass previous layers to dense layer via shortcuts links
            out = layer(*prev_inputs)
            prev_inputs.append(out)
        # the total number of output channels in array prev_inputs is
        # tot_layer * k (because it increases k each dense layer).
        return torch.cat(prev_inputs, 1)


class Transition(nn.Sequential):
    # Transition layer is inserted between dense blocks
    def __init__(self, in_c, out_c):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_c))
        self.add_module('relu', nn.ReLU())
        # the conv layer below will resize the #channels to out_c
        self.add_module('conv', nn.Conv2d(in_c, out_c,
            kernel_size=1, stride=1, bias=False))
        # the pooling layer below will shrink the size by 2
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, n_classes, growth_rate=12,
        block_config=(16, 16, 16), compression=0.5, init_c=24):
        super(DenseNet, self).__init__()

        # assume it is color image (channel=3), build initial layers
        # to convert to channel = init_c.
        if 0: # for small image, do not change the size
            self.add_module('conv0', nn.Conv2d(3, init_c,
                kernel_size=3, stride=1, padding=1, bias=False))
        else: # for larger image, use layers below to reduce the size
            self.add_module('conv0', nn.Conv2d(3, init_c,
                kernel_size=7, stride=2, padding=3, bias=False))
            self.add_module('norm0', nn.BatchNorm2d(init_c))
            self.add_module('relu0', nn.ReLU())
            # MaxPooling output size = (init_c + 2 - 3) / 2 + 1 = init_c / 2
            # (under floor mode)
            self.add_module('pool0', nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, ceil_mode=False
            ))

        # starting from the next layer, we build dense net from dense blocks
        # interleaved by transition layers according to the block_config.
        c = init_c
        for b, n_layers in enumerate(block_config):
            block = DenseBlock(n_layers, c, growth_rate)
            self.add_module('dense-block-%d' % (b + 1), block)
            # predict the output channel size after this block
            c += n_layers * growth_rate
            # append transition layers except the last one block
            if b != len(block_config) - 1:
                # the transition layer will:
                # (1) shrink the number of channels (specified by compression)
                # (2) shrink the output size by 2 due to the pooling layer.
                trans = Transition(c, int(c * compression))
                self.add_module('transition-%d' % (b + 1), trans)
                # update the output channel number
                c = int(c * compression)

        # finally, add a batch norm layer at the end
        self.add_module('norm_final', nn.BatchNorm2d(c))

        # save the number of classes we want to predict
        self.n_classes = n_classes

    def forward(self, inputs):
        # for each layers we have added in dense net ...
        for name, layer in self.named_children():
            # print(inputs.shape)
            # print(layer)
            inputs = layer(inputs)
            # print(inputs.shape)
            # print()
        # termination layers before classifier
        out = F.relu(inputs)
        out = F.avg_pool2d(out, kernel_size=7)
        # flatten the output (inputs.size(0) is the batch size)
        out = out.view(inputs.size(0), -1)
        # use a fully connected classifier here
        classifier = nn.Linear(out.size(1), self.n_classes)
        return classifier(out)

# example for running a denseNet
d1 = torch.rand(480000).reshape(1, 3, 400, 400)
net = DenseNet(10)
out = net(d1)
