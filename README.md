# NSF: Neural Surface Field for Human Modeling from Monocular Depth
#### [Project Page](PlaceHolder) | [Video](PlaceHolder) | [Paper](PlaceHolder)

In ICCV 2023, Paris

[Yuxuan Xue](https://yuxuan-xue.com/)\*, [Bharat Lal Bhatnagar](PlaceHolder)\*, [Riccardo Marin](https://ricma.netlify.app/), [PlaceHolder](PlaceHolder), [PlaceHolder](PlaceHolder), [PlaceHolder](PlaceHolder), [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html) 


Real Virtual Human Group, University of Tübingen & Tübingen AI Center


### 1. Install dependencies

Setup (with [conda](https://docs.conda.io/en/latest/)): 
with `environment.yml` file (if you have old Nvidia GPU, compatible with cuda 11.1):
```
conda env create -f environment.yml
```
or manually (if you have new Nvidia GPU, compatible with cuda 11.8):
```
conda create -n nsf python=3.9
# install pytorch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
# install other dependencies
conda install scipy==1.10.1
pip install open3d
pip install pymeshlab
```
In addition, install `kaolin` from source (https://kaolin.readthedocs.io/en/latest/notes/installation.html) and modify:
```
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.13.0
python setup.py develop
# modify trianglemesh.py to enable different number of faces for multiple subject
# replace kaoin_install_dir/kaolin/kaolin/metrics/trianglemesh.py with kaolin/trianglemesh.py
# make sure that only function point_to_mesh_distance is modified
```
```
### 2. Downlod SMPL model
```
1. Download SMPL (https://smpl.is.tue.mpg.de/) model
```

## Synthetic Data
### BuFF 
1. Download BuFF dataset (https://buff.is.tue.mpg.de/) and unzip it to 
```
DATA_DIR/BuFF/buff_release/subject_id/garment_action/garment_action_frame.ply
```
2. Download BuFF registration (If available) from CAPE (https://cape.is.tue.mpg.de/) and unzip it to
```
DATA_DIR/BuFF/buff_release/subject_id/garment_action/garment_action_frame.npz
```
3. Render RGB image, rasterize depth image, save unprojected colorful point cloud
```
python buff_preprocessing.py
```

## Running
### Learn Fusion Shape via SDF
#### 1. Canonicalize Input Partial Shape

#### 2. Learn Canonical Fusion Shape

### 2. Learn Neural Surface Field based on Fusion Shape