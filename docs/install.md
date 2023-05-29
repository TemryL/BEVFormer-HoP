# Requirements
- python=3.7.7 [GCC 8.4.0]
- pytorch=1.9.1+cu111
- torchvision=0.10.1+cu111
- torchaudio=0.9.1
- mmcv-full=1.4.0
- mmdet=2.14.0
- mmseg=0.14.1
- mdetection3d=0.17.1
- timm=0.6.13
- tensorboard=2.8.0
- torchmetrics=0.11.4


# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**a. Create a virtual environment and activate it.**
```shell
python -m venv --system-site-packages venvs/open-mmlab
source /venvs/open-mmlab/bin/activate
```
**a.1. Upgrade pip.**
```shell
pip install --upgrade pip
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install mmcv-full.**
```shell
pip install --no-cache-dir mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install --no-cache-dir mmdet==2.14.0
pip install --no-cache-dir mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 
python setup.py install
```

**f. Install timm.**
```shell
pip install --no-cache-dir timm
```

**g. Install tensorboard 2.8.0.**
```shell
pip install --no-cache-dir tensorboard==2.8.0
```

**h. Install torchmetrics 0.11.4.**
```shell
pip install --no-cache-dir torchmetrics==0.11.4
```

**i. Clone repo**
```shell
git clone https://github.com/vita-student-projects/BEVFormer_HoP_gr3.git
```


# Detailed Environment Info:
- sys.platform: linux
- Python: 3.7.7 [GCC 8.4.0]
- CUDA available: True
- GPU 0,1: Tesla V100-PCIE-32GB
- NVCC: Cuda compilation tools, release 10.2, V10.2.89
- GCC: gcc (Spack GCC) 8.4.0
- PyTorch: 1.9.1+cu111

- PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.1.2 (Git Hash 98be7e8afa711dc9b66c8ff3504129cb82013cdb)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.1
  - CuDNN 8.0.5
  - Magma 2.5.2
  - Build settings: CUDA_VERSION=11.1, CUDNN_VERSION=8.0.5

- TorchVision: 0.10.1+cu111
- OpenCV: 4.7.0
- MMCV: 1.4.0
- MMCV Compiler: GCC 8.4
- MMCV CUDA Compiler: 10.2
- MMDetection: 2.14.0
- MMSegmentation: 0.14.1
- MMDetection3D: 0.17.1+817468b

# Detailed Packages Infos
- absl-py                 0.7.1
- addict                  2.4.0
- anyio                   3.6.2
- argon2-cffi             21.3.0
- argon2-cffi-bindings    21.2.0
- astunparse              1.6.3
- attrs                   23.1.0
- backcall                0.2.0
- beautifulsoup4          4.12.2
- black                   23.3.0
- bleach                  6.0.0
- Bottleneck              1.2.1
- cachetools              5.3.0
- certifi                 2019.9.11
- cffi                    1.15.1
- chardet                 3.0.4
- click                   8.1.3
- cycler                  0.10.0
- Cython                  0.29.16
- debugpy                 1.6.7
- decorator               5.1.1
- defusedxml              0.7.1
- descartes               1.1.0
- entrypoints             0.4
- exceptiongroup          1.1.1
- fastjsonschema          2.16.3
- filelock                3.12.0
- fire                    0.5.0
- flake8                  5.0.4
- fsspec                  2023.1.0
- gast                    0.3.3
- google-auth             2.18.1
- google-auth-oauthlib    0.4.6
- google-pasta            0.1.8
- grpcio                  1.27.2
- huggingface-hub         0.14.1
- idna                    2.8
- imageio                 2.28.0
- importlib-metadata      6.6.0
- importlib-resources     5.12.0
- iniconfig               2.0.0
- ipykernel               6.16.2
- ipython                 7.34.0
- ipython-genutils        0.2.0
- ipywidgets              8.0.6
- jedi                    0.18.2
- Jinja2                  3.1.2
- joblib                  1.2.0
- jsonschema              4.17.3
- jupyter                 1.0.0
- jupyter_client          7.4.9
- jupyter-console         6.6.3
- jupyter_core            4.12.0
- jupyter-server          1.24.0
- jupyterlab-pygments     0.2.2
- jupyterlab-widgets      3.0.7
- Keras-Preprocessing     1.1.2
- kiwisolver              1.1.0
- llvmlite                0.31.0
- lyft-dataset-sdk        0.0.8
- Markdown                3.4.3
- MarkupSafe              2.1.2
- matplotlib              3.2.2
- matplotlib-inline       0.1.6
- mccabe                  0.7.0
- mistune                 2.0.5
- mmcv-full               1.4.0
- mmdet                   2.14.0
- mmdet3d                 0.17.1
- mmsegmentation          0.14.1
- mpmath                  1.1.0
- mypy-extensions         1.0.0
- nbclassic               0.5.6
- nbclient                0.7.4
- nbconvert               7.3.1
- nbformat                5.8.0
- nest-asyncio            1.5.6
- networkx                2.2
- ninja                   1.11.1
- notebook                6.5.4
- notebook_shim           0.2.3
- numba                   0.48.0
- numexpr                 2.7.0
- numpy                   1.18.5
- nuscenes-devkit         1.1.10
- oauthlib                3.2.2
- opencv-python           4.7.0.72
- opt-einsum              3.1.0
- packaging               23.1
- pandas                  1.0.5
- pandocfilters           1.5.0
- parso                   0.8.3
- pathspec                0.11.1
- pexpect                 4.8.0
- pickleshare             0.7.5
- Pillow                  9.5.0
- pip                     23.1.2
- pkgutil_resolve_name    1.3.10
- platformdirs            3.5.0
- plotly                  5.14.1
- pluggy                  1.0.0
- ply                     3.11
- plyfile                 0.9
- prettytable             3.7.0
- prometheus-client       0.16.0
- prompt-toolkit          3.0.38
- protobuf                3.11.2
- psutil                  5.9.5
- ptyprocess              0.7.0
- pyasn1                  0.5.0
- pyasn1-modules          0.3.0
- pycocotools             2.0.6
- pycodestyle             2.9.1
- pycparser               2.21
- pyflakes                2.5.0
- Pygments                2.15.1
- pyparsing               2.4.2
- pyquaternion            0.9.9
- pyrsistent              0.19.3
- pytest                  7.3.1
- python-dateutil         2.8.2
- pytz                    2019.3
- PyWavelets              1.3.0
- PyYAML                  5.3.1
- pyzmq                   25.0.2
- qtconsole               5.4.2
- QtPy                    2.3.1
- requests                2.22.0
- requests-oauthlib       1.3.1
- rsa                     4.9
- scikit-image            0.19.3
- scikit-learn            1.0.2
- scipy                   1.5.0
- semver                  2.8.1
- Send2Trash              1.8.2
- setuptools              41.2.0
- Shapely                 1.8.5
- six                     1.14.0
- sniffio                 1.3.0
- soupsieve               2.4.1
- sympy                   1.4
- tenacity                8.2.2
- tensorboard             2.8.0
- tensorboard-data-server 0.6.1
- tensorboard-plugin-wit  1.8.1
- termcolor               1.1.0
- terminado               0.17.1
- terminaltables          3.1.10
- threadpoolctl           3.1.0
- tifffile                2021.11.2
- timm                    0.6.13
- tinycss2                1.2.1
- tomli                   2.0.1
- torch                   1.9.1+cu111
- torchaudio              0.9.1
- torchmetrics            0.11.4
- torchvision             0.10.1+cu111
- tornado                 6.2
- tqdm                    4.65.0
- traitlets               5.9.0
- trimesh                 2.35.39
- typed-ast               1.5.4
- typing_extensions       4.5.0
- urllib3                 1.25.6
- virtualenv              16.7.6
- wcwidth                 0.2.6
- webencodings            0.5.1
- websocket-client        1.5.1
- Werkzeug                2.2.3
- wheel                   0.33.4
- widgetsnbextension      4.0.7
- wrapt                   1.11.2
- xarray                  0.14.0
- yapf                    0.33.0
- zipp                    3.15.0
