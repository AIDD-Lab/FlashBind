"""
EGNN Module

This module implements EGNN for representation learning,
based on the egnn-pytorch implementation.
"https://github.com/lucidrains/egnn-pytorch"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from torch.nn import SiLU
from torch import Tensor, einsum
from einops import rearrange

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False
    
    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

def exists(val):
    return val is not None

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask = None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class Attention_Sparse(Attention):
    def __init__(self, **kwargs):
        """ Wraps the attention class to operate with pytorch-geometric inputs. """
        super(Attention_Sparse, self).__init__(**kwargs)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None: 
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1: 
            x, context = map(lambda t: rearrange(t, 'n d -> () n d'), (x, context))
            mask_b = mask.unsqueeze(0) if mask is not None and mask.dim() == 1 else mask
            return self.forward(x, context, mask=mask_b).squeeze() # get rid of batch dim
        # multiple examples in batch - do block-sparse by dense loop
        else:
            x_list = []
            aux_count = 0
            uniques, counts = batch_uniques
            for bi, n_idxs in zip(uniques, counts):
                n = int(n_idxs.item()) if isinstance(n_idxs, torch.Tensor) else int(n_idxs)
                x_list.append( 
                    self.sparse_forward(
                        x[aux_count:aux_count+n], 
                        context[aux_count:aux_count+n],
                        batch_uniques = (bi.unsqueeze(-1), n_idxs.unsqueeze(-1) if isinstance(n_idxs, torch.Tensor) else torch.tensor([n], device=x.device)),
                        mask = mask[aux_count:aux_count+n] if mask is not None else None
                    ) 
                )
                aux_count += n
            return torch.cat(x_list, dim=0)


class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = torch_geometric.nn.norm.LayerNorm(dim)
        self.norm_queries = torch_geometric.nn.norm.LayerNorm(dim)
        self.attn1 = Attention_Sparse(dim=dim, heads=heads, dim_head=dim_head)
        self.attn2 = Attention_Sparse(dim=dim, heads=heads, dim_head=dim_head)

        # can't concat pyg norms with torch sequentials
        self.ff_norm = torch_geometric.nn.norm.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, batch_uniques=None, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x, batch=batch), self.norm_queries(queries, batch=batch)

        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask = mask)
        out     = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        x =  out + res_x
        queries = induced + res_queries

        x_norm = self.ff_norm(x, batch=batch)
        x = self.ff(x_norm) + x_norm
        return x, queries

class EGNN_Sparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and 
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        soft_edge = 0,
        norm_feats = False,
        norm_coors = False,
        norm_coors_scale_init = 1e-2,
        update_feats = True,
        update_coors = True, 
        dropout = 0.,
        coor_weights_clamp_value = None, 
        aggr = "add",
        **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = coor_weights_clamp_value

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1), 
                                         nn.Sigmoid()
        ) if soft_edge else None

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1. 
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        # COORS
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (2, n_edges)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True).clamp(max=1e5)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                                           coors=coors, rel_coors=rel_coors, 
                                                           batch=batch)
        return torch.cat([coors_out, hidden_out], dim=-1)


    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp( torch.cat([x_i, x_j, edge_attr], dim=-1) )
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        
        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                clamp_value = self.coor_weights_clamp_value
                coor_wij.clamp_(min = -clamp_value, max = clamp_value)

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp( torch.cat([hidden_feats, m_i], dim = -1) )
            hidden_out = kwargs["x"] + hidden_out
        else: 
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__) 

class ComplexEncoder(nn.Module):
    """ComplexEncoder for processing protein-ligand complexes."""
    
    def __init__(
        self,
        protein_repr_dim: int = 384,
        ligand_repr_dim: int = 56,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        edge_dim: int = 12,
        m_dim: int = 16,
        dropout: float = 0.1,
        norm_feats: bool = True,
        norm_coors: bool = True,
        fourier_features: int = 0,
        coor_weights_clamp_value: float = 2.0,
        num_edge_types: int = 10,  # Number of different edge types (1-10)
        # Feature dimensions
        atom_type_dim: int = 24,  
        residue_type_dim: int = 24,
        molecule_type_dim: int = 3,
        # Attention controls
        attn_every: int = 0,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        # Noise augmentation
        dist_noise: bool = False,
    ):
        """Initialize ComplexEncoder.
        
        Parameters
        ----------
        protein_repr_dim : int
            Protein representation dimension
        ligand_repr_dim : int
            Ligand representation dimension
        hidden_dim : int
            Hidden dimension for EGNN layers
        output_dim : int
            Output node feature dimension
        num_layers : int
            Number of EGNN layers
        edge_dim : int
            Edge feature dimension
        m_dim : int
            Hidden dimension for edge/node MLPs
        dropout : float
            Dropout rate
        norm_feats : bool
            Whether to normalize features
        norm_coors : bool
            Whether to normalize coordinates
        coor_weights_clamp_value : float
            Clamp value for coordinate weights
        num_edge_types : int
            Number of different edge types
        atom_type_dim : int
            Atom type feature dimension
        residue_type_dim : int
            Residue type feature dimension
        molecule_type_dim : int
            Molecule type one-hot feature dimension
        attn_every : int
            Insert attention after every N EGNN layers (0 disables attention)
        attn_heads : int
            Number of attention heads
        attn_dim_head : int
            Per-head dimension
        """
        super().__init__()
        
        self.protein_repr_dim = protein_repr_dim
        self.ligand_repr_dim = ligand_repr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.num_edge_types = num_edge_types
        self.pos_dim = 3
        self.molecule_type_dim = molecule_type_dim
        # attention config
        self.attn_every = attn_every
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        # noise augmentation
        self.dist_noise = dist_noise
        
        # Projection layers for protein and ligand representations
        self.protein_proj = nn.Linear(protein_repr_dim, hidden_dim)
        self.ligand_proj = nn.Linear(ligand_repr_dim, hidden_dim)
        
        # Define fused feature dimension directly used by EGNN_Sparse
        self.fused_dim = hidden_dim + atom_type_dim + residue_type_dim + self.molecule_type_dim + 1
        
        # EGNN_Sparse layers
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=self.fused_dim,
                pos_dim=self.pos_dim,
                edge_attr_dim=edge_dim,
                m_dim=m_dim,
                dropout=dropout,
                norm_feats=norm_feats,
                norm_coors=norm_coors,
                norm_coors_scale_init=1e-2,
                fourier_features=fourier_features,
                coor_weights_clamp_value=coor_weights_clamp_value,
                aggr="add"
            )
            for _ in range(num_layers)
        ])
        
        # Attention layers (inserted between EGNN layers per attn_every)
        self.attn_layers = nn.ModuleDict()
        if self.attn_every and self.attn_every > 0:
            for i in range(num_layers):
                if ((i + 1) % self.attn_every) == 0:
                    self.attn_layers[str(i)] = GlobalLinearAttention_Sparse(
                        dim=self.fused_dim,
                        heads=self.attn_heads,
                        dim_head=self.attn_dim_head,
                    )
        
        # Output projection expects concatenated coordinates + fused features
        self.output_proj = nn.Linear(self.fused_dim + self.pos_dim, output_dim)
        
        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(num_edge_types + 1, edge_dim)  # +1 for padding (index 0)
        
    def process_sparse_edges(
        self,
        B: int,
        L: int,
        complex_edge_repr: Tensor,
        complex_edge_mask: Tensor,
        complex_mask: Tensor,
    ):
        """Process edge information to create sparse edge_index and edge_attr.
        
        Parameters
        ----------
        B : int
            Batch size
        L : int
            Number of nodes per sample
        complex_edge_repr : Tensor
            Edge representations [B, E, 3] (node_in, node_out, edge_type)
        complex_edge_mask : Tensor
            Edge mask [B, E]
        complex_mask : Tensor
            Node mask [B, L]
            
        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            edge_index [2, total_edges], edge_attr [total_edges, edge_dim], batch [total_nodes]
        """
        device = complex_edge_repr.device

        # Build batch vector and node mapping in a fully vectorized way
        batch_idx, node_idx = torch.where(complex_mask)  # [total_valid_nodes]
        batch = batch_idx  # PyG batch vector aligned with flattened valid nodes

        node_mapping = torch.full((B, L), -1, device=device, dtype=torch.long)
        global_indices = torch.arange(batch_idx.numel(), device=device)
        node_mapping[batch_idx, node_idx] = global_indices

        # Handle case with no valid edges early
        valid_edges_mask = complex_edge_mask.bool()  # [B, E]
        if not valid_edges_mask.any():
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_dim), device=device)
            return edge_index, edge_attr, batch

        # Select valid edges according to edge mask
        E = complex_edge_repr.shape[1]
        bexp = torch.arange(B, device=device).unsqueeze(1).expand(-1, E)  # [B, E]
        valid_b = bexp[valid_edges_mask]                                  # [Ne]
        edge_data = complex_edge_repr[valid_edges_mask]                    # [Ne, 3]
        node_in = edge_data[:, 0].long()
        node_out = edge_data[:, 1].long()
        edge_type = edge_data[:, 2].long()

        # Discard out-of-bounds indices
        in_bounds = (node_in >= 0) & (node_in < L)
        out_bounds = (node_out >= 0) & (node_out < L)
        keep = in_bounds & out_bounds
        if not keep.any():
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_dim), device=device)
            return edge_index, edge_attr, batch
        valid_b = valid_b[keep]
        node_in = node_in[keep]
        node_out = node_out[keep]
        edge_type = edge_type[keep]

        # Discard edges touching masked-out nodes
        keep2 = complex_mask[valid_b, node_in] & complex_mask[valid_b, node_out]
        if not keep2.any():
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_dim), device=device)
            return edge_index, edge_attr, batch
        valid_b = valid_b[keep2]
        node_in = node_in[keep2]
        node_out = node_out[keep2]
        edge_type = edge_type[keep2]

        # Map to flattened global node indices
        flat_in = node_mapping[valid_b, node_in]
        flat_out = node_mapping[valid_b, node_out]

        # Build directed edges (forward + backward) to ensure undirected connectivity
        forward_edges = torch.stack([flat_in, flat_out], dim=0)      # [2, Ne']
        backward_edges = torch.stack([flat_out, flat_in], dim=0)     # [2, Ne']
        edge_index = torch.cat([forward_edges, backward_edges], dim=1)  # [2, 2*Ne']

        # Edge attributes (duplicate for both directions)
        edge_emb = self.edge_type_embedding(edge_type)               # [Ne', edge_dim]
        edge_attr = torch.cat([edge_emb, edge_emb], dim=0)           # [2*Ne', edge_dim]

        return edge_index, edge_attr, batch
    
    def forward(
        self,
        complex_coord: Tensor,
        protein_repr: Tensor,
        ligand_repr: Tensor,
        complex_edge_repr: Tensor,
        complex_edge_mask: Tensor,
        molecule_types: Tensor, 
        atom_types: Tensor,
        residue_types: Tensor,
        residue_indices: Tensor,
        complex_mask: Tensor,
    ):
        """Forward pass.
        
        Parameters
        ----------
        complex_coord : Tensor
            Complex coordinates [B, L, 3]
        protein_repr : Tensor
            Protein representations [B, L_protein, protein_repr_dim] (padded)
        ligand_repr : Tensor
            Ligand representations [B, L_ligand, ligand_repr_dim] (padded)
        complex_edge_repr : Tensor
            Edge representations [B, E, 3]
        complex_edge_mask : Tensor
            Edge mask [B, E]
        molecule_types : Tensor
            Molecule types one-hot [B, L, 3]
        atom_types : Tensor
            Atom types [B, L, atom_type_dim]
        residue_types : Tensor
            Residue types [B, L, residue_type_dim]
        residue_indices : Tensor
            Residue indices scaled to [0, 1] [B, L]
        complex_mask : Tensor
            Complex mask [B, L]
            
        Returns
        -------
        Tensor
            Complex representations [B, L, output_dim]
        """
        # Ensure all inputs are float tensors to match model precision
        complex_coord = complex_coord.float()
        protein_repr = protein_repr.float()
        ligand_repr = ligand_repr.float()
        complex_edge_repr = complex_edge_repr.float()
        complex_edge_mask = complex_edge_mask.bool()
        molecule_types = molecule_types.float()  # one-hot already
        atom_types = atom_types.float()
        residue_types = residue_types.float()
        residue_indices = residue_indices.float()
        complex_mask = complex_mask.bool()
        
        B, L = complex_coord.shape[:2]
        
        # Create masks for different molecule types
        protein_mask = (molecule_types[..., 1] > 0.5) & complex_mask  # [B, L]
        ligand_mask = (molecule_types[..., 2] > 0.5) & complex_mask   # [B, L]

        # Initialize features tensor
        feats = torch.zeros(B, L, self.hidden_dim, device=complex_coord.device, dtype=torch.float32)
        
        # Feature alignment
        if protein_mask.any():
            protein_pos = torch.cumsum(protein_mask, dim=1) - 1
            protein_pos = protein_pos[protein_mask]
            batch_idx = torch.arange(B, device=complex_coord.device).unsqueeze(1).expand_as(protein_mask)[protein_mask]
            protein_feats = self.protein_proj(protein_repr[batch_idx, protein_pos])
            feats[protein_mask] = protein_feats
        
        if ligand_mask.any():
            ligand_pos = torch.cumsum(ligand_mask, dim=1) - 1
            ligand_pos = ligand_pos[ligand_mask]
            batch_idx = torch.arange(B, device=complex_coord.device).unsqueeze(1).expand_as(ligand_mask)[ligand_mask]
            ligand_feats = self.ligand_proj(ligand_repr[batch_idx, ligand_pos])
            feats[ligand_mask] = ligand_feats
        
        # Directly fuse features without MLP
        fused_features = torch.cat([
            feats,
            atom_types,
            residue_types,
            molecule_types,                # one-hot channels
            residue_indices.unsqueeze(-1)  # scalar channel
        ], dim=-1)  # [B, L, fused_dim]
        
        feats = fused_features  # use fused features directly
        
        coors = complex_coord.clone()  # [B, L, 3]
        
        # Add coordinate noise during training (following enzyme_specificity)
        if self.training and self.dist_noise:
            noise = torch.from_numpy(
                np.random.laplace(0.001994, 0.031939, coors.shape)
            ).float().to(coors.device)
            coors = coors + noise
        
        # Process sparse edges
        edge_index, edge_attr, batch = self.process_sparse_edges(
            B, L, complex_edge_repr, complex_edge_mask, complex_mask
        )
        
        # Convert to PyG format: extract only valid nodes
        valid_coors = coors[complex_mask]  # [total_valid_nodes, 3]
        valid_feats = feats[complex_mask]   # [total_valid_nodes, fused_dim]
        
        # Combine coordinates and features for PyG input
        x = torch.cat([valid_coors, valid_feats], dim=-1)  # [total_valid_nodes, 3+fused_dim]

        # Apply EGNN_Sparse layers
        for i, layer in enumerate(self.egnn_layers):
            x = layer(x, edge_index, edge_attr, batch)
            # optional attention after certain layers
            if self.attn_every and self.attn_every > 0 and (((i + 1) % self.attn_every) == 0):
                if str(i) in self.attn_layers:
                    feats_only = x[:, self.pos_dim:]
                    feats_out, _ = self.attn_layers[str(i)](feats_only, feats_only, batch=batch)
                    x = torch.cat([x[:, :self.pos_dim], feats_out], dim=-1)
            
            if torch.cuda.is_available():
                try:
                    reserved = torch.cuda.memory_reserved()
                    allocated = torch.cuda.memory_allocated()
                    threshold_bytes = int(getattr(self, "cuda_cache_fragment_threshold_bytes", 3.5 * 1024 ** 3))
                    if reserved - allocated > threshold_bytes:
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        
        # Extract coordinates and features
        updated_coors = x[:, :self.pos_dim]  # [total_valid_nodes, 3]
        updated_feats = x[:, self.pos_dim:]  # [total_valid_nodes, fused_dim]
        
        # Restore to batch format
        output_coors = torch.zeros_like(coors)  # [B, L, 3]
        output_feats = torch.zeros_like(feats)  # [B, L, fused_dim]
        
        output_coors[complex_mask] = updated_coors
        output_feats[complex_mask] = updated_feats
        
        output_feats = torch.cat([output_coors, output_feats], dim=-1)

        # Output projection
        output = self.output_proj(output_feats)  # [B, L, output_dim]
        
        return output 