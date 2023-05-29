# Pretrain HoP only on 2 GPUs
```
./tools/dist_train.sh ./projects/configs/bevformer_hop/bevformer_tiny_hop_only.py 2
```

# Train BEVFormer with HoP on 2 GPUs
```
./tools/dist_train.sh ./projects/configs/bevformer_hop/bevformer_tiny_hop_bi_loss.py 2
```

# Test with 2 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_tiny.py ./ckpts/bevformer_tiny_epoch_24.pth 2
```