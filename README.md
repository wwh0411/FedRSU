# FedRSU: Federated Learning for Scene Flow Estimation on Roadside Units

This is the official repository for our work "[FedRSU: Federated Learning for Scene Flow Estimation on Roadside Units]"(https://arxiv.org/abs/2401.12862)

## **Installation**

```shell
conda create --name fedrsu python=3.7 cmake=3.22.1

# torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# pypcd
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install

# pointnet2
cd lib/pointnet2
python3 setup.py install
cd ../../

# others
opencv-python-headless, open3d, matplotlib, tensorboardx, pyyaml, pyquaternion
```

## **Datasets**
Our dataset is now available through:
* [Google Drive](https://drive.google.com/file/d/1At3tG0kZHrJnTEGM95NDCcAcnhRntPJx/view?usp=drive_link) 
* [Baidu Netdisk](https://pan.baidu.com/s/16Qg6xUmvPk9gPXWm366Vtw) (提取码: 56js)

Download and put the unzipped four directories in `./data`.

The dataset downloaded here is a zip file composed of four separate directories: 
`Dair-V2X`, `LUMPI`, `IPS300+` and `Campus`.
Due to acknowledgement reasons, we haven't include `IPS300+` in this initial version.
We will update it in the next version soon.

## **Training**
```bash
conda activate fedrsu
```
For non-distibuted setting: (`Recommended`)
```bash
> single gpu
python train.py --config ./config/example.yaml --alg fedavg --gpu 0
```
For distributed / Multi-GPU setting:
```bash
> multiple gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --config ./example_ddp.yaml --ddp --alg fedavg --gpu 0

```

## **Citation**
```
@article{fang2024fedrsu,
  title={FedRSU: Federated Learning for Scene Flow Estimation on Roadside Units},
  author={Fang, Shaoheng and Ye, Rui and Wang, Wenhao and Liu, Zuhong and Wang, Yuxiao and Wang, Yafei and Chen, Siheng and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2401.12862},
  year={2024}
}
```