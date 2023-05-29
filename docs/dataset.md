

## NuScenes
Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download).
To save exporatation time, we have made the compressed blobs of nuScenes V1.0 full dataset available on SCITAS at `/work/vita/datasets/bev/data/nuscenes_full_compressed` 


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*BEVFormer repo generates custom annotation files which are different from mmdet3d's*

```
python tools/prepare_data.py nuscenes --version v1.0 --root-path ./data/nuscenes --out-dir ./data/nuscenes --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.
To save time, we have made `nuscenes_infos_temporal_{train,val}.pkl` available [HERE](https://www.nuscenes.org/download). To use them, place them according to the folder structure bellow and update the file with your `data_root_path` by running:

```
python update_nuscenes_infos.py data/nuscenes/nuscenes_infos_temporal_train.pkl data_root_path
python update_nuscenes_infos.py data/nuscenes/nuscenes_infos_temporal_val.pkl data_root_path

```

**Folder structure**
```
BEVFormer_HoP_gr3
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```