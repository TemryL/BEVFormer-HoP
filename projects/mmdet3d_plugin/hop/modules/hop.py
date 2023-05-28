import torch
import torch.nn as nn 
from .temporal_decoder import TemporalDecoder
from .object_decoder import ObjectDecoder


class HoP(nn.Module):
    def __init__(self, hop_pred_idx=1, history_length=5, embed_dims=256, bev_h=200, bev_w=200, num_classes=10):
        '''
        Args:
            hop_pred_idx (int): index of the prediction BEV feature in the input sequence
            history_length (int): number of historical BEV features, including the current BEV feature
        '''
        super().__init__()
        self.k = hop_pred_idx
        self.N = history_length
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.temp_embedding = nn.Embedding(history_length-1, embed_dims)
        self.temporal_decoder = TemporalDecoder(embed_dims, bev_h, bev_w)
        self.object_decoder = ObjectDecoder(embed_dims, bev_h, bev_w, num_classes)
    
    def forward(self, bev_history):
        '''Forward pass of the Historical Object Prediction (HoP) framework
        Args:
            bev_history (list(Tensor)): BEV feature sequence (first element at time t, second at time t-1 ...) that consists of N historical BEV features and the current BEV feature, each element with shape (bs, bev_h*bev_w, embed_dims)
        Returns:
            outs: output of the object decoder in same format than the BEVFormerHead
        ''' 
        dtype = bev_history[0].dtype
        device = bev_history[0].device
        
        # Add temporal positional embeddings to BEV feature maps
        for t, bev_features in enumerate(bev_history):
            temporal_pos_embeds = self.temp_embedding.weight.to(dtype).to(device)
            temporal_pos_embeds = temporal_pos_embeds.unsqueeze(1).expand(-1, self.bev_h*self.bev_w, -1)
            bev_history[t] = bev_features + temporal_pos_embeds
    
        bev_history.reverse()
        B_adj = [bev_history[self.k - 1], bev_history[self.k + 1]]  # adjacent BEV features at time t-k-1 and t-k+1
        bev_history.pop(self.k)                                     # remove BEV feature at time t-k
        B_rem = bev_history                                         # remaining BEV features
        
        B_pred = self.temporal_decoder(B_adj, B_rem)       # reconstructed BEV feature at time t-k
        outs = self.object_decoder(B_pred)                 # 3D predictions
        
        return outs

# if __name__ == '__main__':
#     B = torch.randn([1, 50*50, 256])
#     bev_history = [B, B, B, B, B]
    
#     hop = HoP(hop_pred_idx=1, history_length=5, embed_dims=256, bev_h=50, bev_w=50)
#     outs = hop(bev_history)
#     print(outs)