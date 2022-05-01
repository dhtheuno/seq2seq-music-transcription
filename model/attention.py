import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)
        attention_score = torch.bmm(query, value.transpose(1,2))
        attention_distribution = F.softmax(
            attention_score.view(-1, input_size), dim=1).view(
                batch_size, -1, input_size)
        context = torch.bmm(attention_distribution, value)
        return context, attention_distribution
        