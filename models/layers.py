import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

class GAT(nn.Module):

    def __init__(self, d_node, d_edge, d_inner, n_heads, dropout) -> None:
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_inner = d_inner
        self.n_heads = n_heads

        self.node_proj = nn.Linear(self.d_node, self.n_heads * self.d_inner, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_inner))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_inner))
        self.node_out = nn.Linear(self.d_inner, self.d_node)

        self.edge_proj = nn.Linear(self.d_edge, self.n_heads * self.d_inner, bias=False)
        self.edge_out = nn.Linear(self.d_inner, self.d_edge)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feature, edge_feature, adj_mask):

        # node_feature (n, d_node)
        # edge_feature (n, n, d_edge)
        # adj_mask (n, n)
        n_node = len(node_feature)
        # process node
        node_in = self.dropout(node_feature) 
        node_projection = self.node_proj(node_in).reshape(-1, self.n_heads, self.d_inner) # (n, n_heads, d_inner)
        node_projection = self.dropout(node_projection)
        source_node = node_projection * self.scoring_fn_source # (n, n_heads, d_inner)
        target_node = node_projection * self.scoring_fn_target # (n, n_heads, d_inner)
        scores_source = source_node.unsqueeze(1) # (n, 1, n_heads, d_inner)
        scores_target = target_node.unsqueeze(0) # (1, n, n_heads, d_inner)
        score_matrix = scores_source + scores_target # (n, n, n_heads, d_inner)
        # process edge
        edge_peojection = self.edge_proj(edge_feature).reshape(n_node, -1, self.n_heads, self.d_inner) # (n, n, n_heads, d_inner)
        score_matrix += edge_peojection # (n, n, n_heads, d_inner)
        edge_out = self.edge_out(torch.sum(score_matrix, dim=2)) # (n, n, d_edge)
        # calculate attention
        all_scores = torch.sum(score_matrix, dim=-1) # (n, n, n_head)
        all_scores.masked_fill((adj_mask==0), -1e9)
        attention = torch.softmax(all_scores, dim=1) 
        node_out = torch.bmm(attention.permute(2,0,1),node_projection.transpose(0,1)) # (n_heads, n, d_inner)
        node_out = torch.sum(node_out, dim=0)
        node_out = self.node_out(node_out)

        return node_feature + node_out, edge_feature + edge_out, adj_mask

class CrossGAT(nn.Module):
    def __init__(self, d_sub_node, d_prot_node, d_inner, dropout) -> None:
        super().__init__()
        self.sub_proj = nn.Linear(d_sub_node, d_inner)
        self.prot_proj = nn.Linear(d_prot_node, d_inner)
        self.sub_out = nn.Linear(d_inner, d_sub_node)
        self.prot_out = nn.Linear(d_inner, d_prot_node)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot_node, prot_idx, sub_node, sub_idx):
        prot_proj = self.dropout(self.prot_proj(prot_node))
        sub_proj = self.dropout(self.sub_proj(sub_node))
        prot_int = torch.index_select(prot_proj, 0, prot_idx)
        sub_int = torch.index_select(sub_proj, 0, sub_idx)
        scores = prot_int.unsqueeze(1) + sub_int.unsqueeze(0)
        scores = torch.sum(scores, 2)
        prot_scores = torch.softmax(scores, 1)
        sub_scores = torch.softmax(scores, 0).transpose(0,1)
        prot_out = torch.bmm(prot_scores, sub_proj)
        sub_out = torch.bmm(sub_scores, prot_proj)
        prot_out = self.prot_out(prot_out)
        sub_out = self.sub_out(sub_out)

        return prot_node+prot_out, prot_idx, sub_node+sub_out, sub_idx



class GCNN(nn.Module):

    def __init__(self, d_sub_node, d_prot_node, d_proj, kernel, dropout):
        super().__init__()
        self.kernel = kernel
        self.d_proj = d_proj
        self.prot_proj = nn.Linear(d_prot_node, d_proj)
        self.sub_proj = nn.Linear(d_sub_node, d_proj)
        self.attention_proj = nn.Linear(2*d_proj, 1)
        self.sub_out = nn.Parameter(torch.Tensor(d_proj, d_sub_node))
        self.prot_out = nn.Parameter(torch.Tensor(d_proj, d_prot_node))
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot_node, sub_node, int_index):
        prot_proj = self.prot_proj(prot_node) # (n, d_proj)
        reaction_potion = torch.index_select(sub_node, 0, int_index).copy
        reaction_potion = torch.sum(self.sub_proj(reaction_potion), 0) # (1, d_proj)
        num_windows = len(prot_node)-self.kernel+1
        scores = torch.tensor(num_windows, self.d_proj)
        for i in range(num_windows):
            window = prot_proj[i:i+self.kernel-1] # (kernel, d_proj)
            score = torch.cat((window, reaction_potion.expand_as(window)), dim=1) # (kernel, 2*d_proj)
            score = self.dropout(score)
            attention = torch.softmax(self.attention_proj(score),dim=0).reshape(1,self.kernel) # (1. kernel)
            local_score = torch.matmul(attention, window) # (1, d_proj)
            scores[i] = local_score + reaction_potion
        rank = torch.sum(scores, dim=1)
        top_window = torch.argmax(rank)
        sub_node[int_index] += torch.matmul(score[top_window].copy(), self.sub_out)
        prot_node[window: window+self.kernel-1] += torch.matmul(score[top_window].copy(), self.prot_out)
        return prot_node, sub_node, int_index

class RepeatModule(nn.Module):

    def __init__(self, d_prot_node, d_sub_node, d_prot_edge, d_sub_edge, d_inner, n_heads, kernel, dropout) -> None:
        super().__init__()
        self.prot_gat = GAT(d_prot_node, d_prot_edge, d_inner, n_heads, dropout)
        self.sub_gat = GAT(d_sub_node, d_sub_edge, d_inner, n_heads, dropout)
        self.int_layer = CrossGAT(d_sub_node, d_prot_node, d_inner, kernel, dropout)

    def forward(self, data):
        prot_node, prot_edge, prot_adj, sub_node, sub_edge, sub_adj, prot_idx, sub_idx = data
        prot_node, prot_edge, prot_adj = self.prot_gat(prot_node, prot_edge, prot_adj)
        sub_node, sub_edge, sub_adj = self.sub_gat(sub_node, sub_edge, sub_adj)
        prot_node, sub_node, sub_idx = self.int_layer(prot_node, prot_idx, sub_node, sub_idx)

        return (prot_node, prot_edge, prot_adj, sub_node, sub_edge, sub_adj, sub_idx)

class FNN(nn.Module):

    def __init__(self, d_sub_node, d_fnn, out_dim, dropout) -> None:
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(d_sub_node, d_fnn), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(d_fnn, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)