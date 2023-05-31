import copy
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import _load_checkpoint_with_prefix, load_state_dict


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ObjectDecoder(nn.Module):
    """
    Object decoder module
    """

    def __init__(self, embed_dims, bev_h, bev_w, num_classes, num_query=900):
        """
        Args:
            embed_dims (int): number of embedding dimensions
            bev_h (int): height of BEV feature maps
            bev_w (int): width of BEV feature maps
            num_classes (int): number of classes
            num_query (int): number of queries
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.num_query = num_query
        self.num_reg_fcs = 2
        self.cls_out_channels = num_classes
        self.code_size = 10
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        # Load pretrained object decoder and freeze layers
        self.decoder = build_transformer_layer_sequence(
            dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=embed_dims,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=embed_dims,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=2 * embed_dims,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            )
        )
        state_dict = _load_checkpoint_with_prefix(
            prefix="pts_bbox_head.transformer.decoder",
            filename="ckpts/bevformer_r101_dcn_24ep.pth",
        )
        load_state_dict(self.decoder, state_dict)
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()

        # Initialize query embedding and reference points
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, 3)

        # Initialize classification and regression heads
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        num_pred = self.decoder.num_layers
        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)

    def forward(self, bev_embed):
        """Forward function.
        Args:
            bev_embed (Tensor): BEV features with shape (bs, bev_h*bev_w, embed_dims)
        Returns:
            outs (dict): Dict of outputs
                bev_embed (Tensor): BEV features with shape (bs, bev_h*bev_w, embed_dims)
                all_cls_scores: Classification scores
                all_bbox_preds: Bounding box predictions
        """

        bs = bev_embed.size(0)
        dtype = bev_embed.dtype

        object_query_embeds = self.query_embedding.weight.to(dtype)
        query_pos, query = torch.split(object_query_embeds, self.embed_dims, dim=1)

        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=self.reg_branches,
            spatial_shapes=torch.tensor(
                [[self.bev_h, self.bev_w]], device=query.device
            ),
            level_start_index=torch.tensor([0], device=query.device),
        )

        inter_references_out = inter_references
        outputs = bev_embed, inter_states, init_reference_out, inter_references_out

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs
