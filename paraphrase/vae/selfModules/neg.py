import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

from utils.functional import *


class NEG_loss(nn.Module):
    def __init__(self, num_classes, embed_size):
        super(NEG_loss, self).__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.out_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))

        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.in_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))

    def forward(self, input_labes, out_labels, num_sampled):

        use_cuda = self.out_embed.weight.is_cuda

        [batch_size] = input_labes.size()

        input = self.in_embed(input_labes)
        output = self.out_embed(out_labels)

        noise = Variable(t.Tensor(batch_size, num_sampled).uniform_(0, self.num_classes - 1).long())
        if use_cuda:
            noise = noise.cuda()
        noise = self.out_embed(noise).neg()

        log_target = (input * output).sum(1).squeeze().sigmoid().log()

        sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        loss = log_target + sum_log_sampled

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()
