import torch
import torch.nn as nn
import torch.nn.functional as F


class CiSTAGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_nodes, num_heads, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, seq_len, hidden_dim))
        self.node_embedding = nn.Parameter(torch.randn(1, num_nodes, 1, hidden_dim))

        self.t_encoder = TemporalEncoder(hidden_dim, num_heads, dropout)
        self.s_encoder = SpatioEncoder(hidden_dim, dropout)

        self.t_encoder_2 = TemporalEncoder(hidden_dim, num_heads, dropout)
        self.s_encoder_2 = SpatioEncoder(hidden_dim, dropout)

        self.dag_generator = DAGGNN(hidden_dim, num_nodes=num_nodes)
        self.gcn = GCNLayer(hidden_dim, hidden_dim)

        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, X, mode="train"):
        loss_dag = torch.tensor(0.0, device=A.device)
        loss_sparse = torch.tensor(0.0, device=A.device)
        loss_diff = torch.tensor(0.0, device=A.device)

        h = self.input_proj(X) + self.pos_embedding + self.node_embedding

        h1 = self.t_encoder(A, h)

        if mode in ["train", "test_direct"]:
            # Spatio-Temporal Encoding
            h_att = self.s_encoder(A, h1)
            h_att = self.t_encoder_2(A, h_att)
            h_att = self.s_encoder_2(A, h_att)[:, :, -1, :]

            h2, causal_adj = self.dag_generator(h1)

            h_out = F.relu(h_att + self.gcn(h_att, causal_adj))
            h_out = torch.matmul(causal_adj, h_out)

            loss_dag = self._dag_loss(self.dag_generator.adj_A)
            loss_sparse = torch.norm(self.dag_generator.adj_A, p=1)

        elif mode == "pretrain":
            # Only when using CISTA_2
            h2, causal_adj = self.dag_generator(h1)
            h_out = F.relu(h2 + self.gcn(h2, causal_adj))
            h_out = torch.matmul(causal_adj, h_out)

            loss_dag = self._dag_loss(self.dag_generator.adj_A)
            loss_sparse = torch.norm(self.dag_generator.adj_A, p=1)

        elif mode in ["finetune", "test"]:
            h2, _ = self.dag_generator(h1)

            h_att = self.s_encoder(A, h1)
            h_att = self.t_encoder_2(A, h_att)
            h_att = self.s_encoder_2(A, h_att)[:, :, -1, :]

            # InfoNCE Loss
            loss_diff = infonce_loss(h_att.mean(dim=1), h2.mean(dim=1))
            h_out = h_att

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Weighted Structural Loss
        exp = int(A.shape[-1] // 10)
        total_structure_loss = (pow(0.1, exp) * loss_dag) + (1e-5 * loss_sparse) + (0.01 * loss_diff)

        h_out = F.relu(self.fc_1(h_out))
        output = self.fc_2(h_out)

        return self.dropout(output), total_structure_loss

    def _dag_loss(self, adj):
        return torch.trace(torch.matrix_exp(adj * adj)) - adj.shape[0]


class DAGGNN(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        init_adj = 0.5 * torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        self.adj_A = nn.Parameter(init_adj)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, h):
        weights = F.softmax(nn.Parameter(torch.randn(h.shape[2], device=h.device)), dim=0)
        h_agg = (h * weights.view(1, 1, -1, 1)).sum(dim=2).double()  # Use double per original paper

        adj_sinh = torch.sinh(3.0 * self.adj_A)

        I = torch.eye(adj_sinh.shape[0], device=h.device).double()
        adj_inv = torch.inverse(I - adj_sinh.T)

        mean_feat = torch.matmul(adj_inv, torch.mean(torch.matmul(I - adj_sinh.T, h_agg), dim=0))
        logits = torch.matmul(I - adj_sinh.T, h_agg - mean_feat)

        out = self.decoder(logits.float())
        adj_soft = F.softmax(adj_sinh, dim=-1).unsqueeze(0).repeat(h.shape[0], 1, 1)

        return out, adj_soft.float()


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attention = TemporalAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, X):
        out = self.attention(X)
        return self.norm(X + self.dropout(out))


class SpatioEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = GraphAttention(hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, A, X):
        out = self.attention(X, A)
        return self.norm(X + out)


class TemporalAttention(nn.Module):
    def __init__(self, hid_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads
        self.qkv = nn.Linear(hid_dim, 3 * hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, L, D = x.shape
        qkv = self.qkv(x).reshape(B * N, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, L, D)
        return self.dropout(self.fc(out))


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        # h: (B, N, L, D) -> (B*L, N, D)
        B, N, L, D = h.shape
        h_reshaped = h.transpose(1, 2).reshape(B * L, N, D)

        # Normalize Adjacency
        I = torch.eye(N, device=adj.device)
        d_inv_sqrt = torch.diag(torch.pow(adj.sum(1) + 1, -0.5))
        adj_norm = d_inv_sqrt @ (adj + I) @ d_inv_sqrt

        Wh = self.W(h_reshaped)  # (B*L, N, D)

        Wh_rep_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_rep_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        a_input = torch.cat([Wh_rep_i, Wh_rep_j], dim=-1)

        e = self.leakyrelu(self.a(a_input)).squeeze(-1)  # (B*L, N, N)


        attention = F.softmax(e, dim=-1)

        h_prime = torch.bmm(attention, Wh)
        h_prime = h_prime.view(B, L, N, D).transpose(1, 2)

        return F.elu(h_prime)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A):
        # A: (B, N, N)
        I = torch.eye(A.shape[1], device=A.device).unsqueeze(0)
        A_hat = A + I
        D_inv_sqrt = torch.diag_embed(torch.pow(A_hat.sum(dim=2), -0.5))
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return self.linear(A_norm @ x)


def infonce_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.T
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim / temperature, labels)