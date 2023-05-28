import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention, build_positional_encoding


class TemporalDecoder(nn.Module):
    def __init__(self, embed_dims, bev_h, bev_w):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.short_term_decoder = ShortTermTemporalDecoder(embed_dims, bev_h, bev_w)
        self.long_term_decoder = LongTermTemporalDecoder(embed_dims, bev_h, bev_w)
        self.features_fusion = nn.Conv2d(embed_dims + int(embed_dims/self.long_term_decoder.reduction), embed_dims, (3,3), padding='same')

    def forward(self, B_adj, B_rem):
        '''Forward pass of the temporal decoder
        Args:
            B_adj (list(Tensor)): adjacent BEV feature (len(B_adj)=2), each element with shape (bs, bev_h*bev_w, embed_dims)
            B_rem (list(Tensor)): remaining BEV features (len(B_rem)=N-3), each element with shape (bs, bev_h*bev_w, embed_dims)
        Returns:
            B_pred (Tensor): reconstructed BEV feature at time t-k with shape (bs, bev_h*bev_w, embed_dims)
        '''
        
        B_short = self.short_term_decoder(B_adj)   # apply short-term temporal decoder
        B_long = self.long_term_decoder(B_rem)     # apply long-term temporal decoder
        B_pred = torch.cat([B_short, B_long], 2)   # concatenate short-term and long-term temporal information
        
        # Perform feature fusion by 3×3 convolution.
        B_pred = B_pred.permute(0, 2, 1)
        B_pred = B_pred.view(-1, B_pred.shape[1], self.bev_h, self.bev_w)
        B_pred = self.features_fusion(B_pred)
        B_pred = B_pred.view(-1, B_pred.shape[1], self.bev_h*self.bev_w)
        B_pred = B_pred.permute(0, 2, 1)
        
        return B_pred


class ShortTermTemporalDecoder(nn.Module):
    def __init__(self, embed_dims, bev_h, bev_w):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        self.attn_cfg = dict(
            type='MultiScaleDeformableAttention',
            embed_dims=self.embed_dims,
            num_points=8,
            num_levels=1
        )
        self.ffn_cfg=dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.,
            act_cfg=dict(type='ReLU', inplace=True)
        )
        
        self.deform_attn = build_attention(self.attn_cfg)
        self.ffn = build_feedforward_network(self.ffn_cfg)
        
        self.query_embed = nn.Embedding(self.bev_h*self.bev_w, self.embed_dims)
        self.reference_points = nn.Linear(self.embed_dims, 2)
    
    def forward(self, B_adj):
        '''Forward pass of the short-term temporal decoder
        Args:
            B_adj (list(Tensor)): adjacent BEV feature (len(B_adj)=2), each element with shape (bs, bev_h*bev_w, embed_dims)
        Returns:
            B_short (Tensor): short-term temporal information with shape (bs, bev_h*bev_w, embed_dims)
        '''
        bs = B_adj[0].size(0)
        dtype = B_adj[0].dtype
        device = B_adj[0].device
        B_short = torch.zeros([bs, self.bev_h*self.bev_w, self.embed_dims]).to(device)

        Q_short = self.query_embed.weight.to(dtype).to(device)
        Q_short = Q_short.unsqueeze(1).repeat(1, bs, 1)
        
        reference_points = self.reference_points(Q_short).to(device)
        reference_points = reference_points.sigmoid()
        reference_points = reference_points.permute(1, 0, 2)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, self.attn_cfg['num_levels'], 1)
        
        for V in B_adj:
            V = V.permute(1, 0, 2)

            B_short += self.deform_attn(
                query=Q_short,
                key=None,
                value=V,
                reference_points=reference_points,
                spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=Q_short.device),
                level_start_index=torch.tensor([0], device=Q_short.device)
            ).permute(1, 0, 2)
        
        B_short = self.ffn(B_short)
        
        return B_short


class LongTermTemporalDecoder(nn.Module):
    def __init__(self, embed_dims, bev_h, bev_w, reduction=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.reduction = reduction
        
        self.attn_cfg = dict(
            type='MultiScaleDeformableAttention',
            embed_dims=int(self.embed_dims/self.reduction),
            num_points=8,
            num_levels=1
        )
        self.ffn_cfg=dict(
            type='FFN',
            embed_dims=int(self.embed_dims/self.reduction),
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.,
            act_cfg=dict(type='ReLU', inplace=True)
        )
        
        self.deform_attn = build_attention(self.attn_cfg)
        self.ffn = build_feedforward_network(self.ffn_cfg)
        
        self.query_embed = nn.Embedding(self.bev_h*self.bev_w, int(self.embed_dims/self.reduction))
        self.reference_points = nn.Linear(int(self.embed_dims/self.reduction), 2)
        self.reduction_layer = nn.Linear(self.embed_dims, int(self.embed_dims/self.reduction))
    
    def forward(self, B_rem):
        '''Forward pass of the long-term temporal decoder
        Args:
            B_rem (list(Tensor)): remaining BEV features (len(B_rem)=N-3), each element with shape (bs, bev_h*bev_w, embed_dims)
        Returns:
            B_long (Tensor): long-term temporal information with shape (bs, bev_h*bev_w, embed_dims)
        '''
        bs = B_rem[0].size(0)
        dtype = B_rem[0].dtype
        device = B_rem[0].device
        B_long = torch.zeros([bs, self.bev_h*self.bev_w, int(self.embed_dims/self.reduction)]).to(device)
        
        Q_long = self.query_embed.weight.to(dtype).to(device)
        Q_long = Q_long.unsqueeze(1).repeat(1, bs, 1)
        
        reference_points = self.reference_points(Q_long).to(device)
        reference_points = reference_points.sigmoid()
        reference_points = reference_points.permute(1, 0, 2)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, self.attn_cfg['num_levels'], 1)
        
        for V in B_rem:
            V = V.permute(1, 0, 2)
            V = self.reduction_layer(V)
            
            B_long += self.deform_attn(
                query=Q_long,
                key=None,
                value=V,
                reference_points=reference_points,
                spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=Q_long.device),
                level_start_index=torch.tensor([0], device=Q_long.device)
            ).permute(1, 0, 2)
        
        B_long = self.ffn(B_long)
        
        return B_long


# if __name__ == '__main__':
    # B = torch.randn([1, 50*50, 256])
    
    # B_adj = [B, B]
    # B_rem = [B, B, B, B, B]
    
    # temporal_decoder = TemporalDecoder(256, 50, 50)
    # B_pred = temporal_decoder(B_adj, B_rem)