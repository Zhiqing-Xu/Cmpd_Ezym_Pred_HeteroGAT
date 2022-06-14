from turtle import forward
import torch
from torch import nn
from layers import RepeatModule, FNN

class Int_GCN(nn.Module):

    def __init__(self, n_layers, d_prot_node, d_sub_node, d_prot_edge, d_sub_edge, d_inner, n_heads, kernel, dropout, d_fnn) -> None:
        super().__init__()
        self.main = nn.ModuleList()
        for _ in range(n_layers):
            self.main.append(RepeatModule(d_prot_node, d_sub_node, d_prot_edge, d_sub_edge, d_inner, n_heads, kernel, dropout))
        self.FNN = FNN(d_sub_node, d_fnn, 1, dropout)

    def forward(self, data):
        for layer in self.main:
            data = layer(data)
        data = self.main(data)
        prot_node, prot_edge, prot_adj, sub_node, sub_edge, sub_adj, sub_idx = data
        reaction_group = torch.sum(torch.index_select(sub_node,0,sub_idx),0)
        output = self.FNN(reaction_group)

        return output