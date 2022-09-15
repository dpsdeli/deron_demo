import torch
from torch import nn
from torch.nn import functional as F


class TripletLoss(nn.Module):
    def __init__(self, num_neg, margin=1, smooth=False, average=True, device='cpu'):
        super().__init__()
        self.margin = torch.FloatTensor([margin]).to(device)
        self.average = average
        self.smooth = smooth
        self.num_neg = num_neg

    def forward(self, anchor_features, sample_features, postive_idx):
        """
            postive_idx     (torch.LongTensor): one-hot label
        """
        batch_size = anchor_features.shape[0]
        distances = 1. - torch.matmul(anchor_features, torch.t(sample_features))
        losses = 0
        for dist, pos in zip(distances, postive_idx):
            pos_mask = pos.bool()
            pos_dist = torch.masked_select(dist, pos_mask)

            if self.num_neg:
                # Random select negative samples
                idx = torch.multinomial(1.-pos, num_samples=self.num_neg)
                neg_dist = dist[idx]
            else:
                neg_dist = torch.masked_select(dist, ~pos_mask)

            num_pos, num_neg = pos_dist.shape[0], neg_dist.shape[0]
            if self.smooth:
                sum_p = torch.sum(torch.exp(pos_dist), dim=0)
                sum_n = torch.sum(torch.exp(-neg_dist), dim=0)
                losses += torch.log(sum_p * sum_n + 1.)

            else:
                pos_dist = pos_dist.unsqueeze(1).repeat(1, num_neg)
                neg_dist = neg_dist.unsqueeze(0).repeat(num_pos, 1)
                margin = self.margin.expand_as(pos_dist)
                losses += F.relu(pos_dist - neg_dist + margin).mean()

        return losses/batch_size if self.average else losses
