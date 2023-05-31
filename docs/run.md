# Run and Eval

## Pretrain HoP using 2 GPUs
```
./tools/dist_train.sh ./projects/configs/bevformer_hop/bevformer_tiny_hop_only.py 2
```

## Train BEVFormer with HoP (bi-loss weight 0.25) using 2 GPUs
```
./tools/dist_train.sh ./projects/configs/bevformer_hop/bevformer_tiny_hop_bi_loss_025.py 2
```

## Test original BEVFormer tiny version using 2 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_tiny.py ./ckpts/bevformer_tiny_epoch_24.pth 2
```

## Test enhanced BEVFormer tiny version with HoP (bi-loss weight 0.25) using 2 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer_hop/bevformer_tiny_hop_bi_loss_025.py ./ckpts/bevformer_tiny_hop_bi_loss_025_epoch_2.pth 2
```