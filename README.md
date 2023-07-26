# NSF: Neural Surface Field for Human Modeling from Monocular Depth
#### [Project Page](PlaceHolder) | [Video](PlaceHolder) | [Paper](PlaceHolder)

In ICCV 2023, Paris

[Yuxuan Xue](https://yuxuan-xue.com/)\*, [Bharat Lal Bhatnagar](PlaceHolder)\*, [Riccardo Marin](https://ricma.netlify.app/), [PlaceHolder](PlaceHolder), [PlaceHolder](PlaceHolder), [PlaceHolder](PlaceHolder), [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html) 


Real Virtual Human Group, University of Tübingen & Tübingen AI Center

## Background
### Install dependencies

Setup (with [conda](https://docs.conda.io/en/latest/)): 
with `environment.yml` file (if you have old Nvidia GPU, compatible with cuda 11.1):
```
conda env create -f environment.yml
```
or manually (if you have new Nvidia GPU, compatible with cuda 11.7):
```
conda create -n nsf python=3.9
conda activate nsf
# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
# install other dependencies
pip install scipy==1.10.1
pip install open3d
pip install pymeshlab
pip install chumpy
pip install opencv-python
python -m pip install numpy==1.23.1
pip install trimesh
python -m pip install -U scikit-image
```
In addition, install `kaolin` from source (https://kaolin.readthedocs.io/en/latest/notes/installation.html) and modify:
```
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.13.0
python setup.py develop
# modify trianglemesh.py to enable different number of faces for multiple subject
# replace KAOLIN_DIR/kaolin/kaolin/metrics/trianglemesh.py with kaolin/trianglemesh.py
# make sure that only function point_to_mesh_distance is modified
```

### Downlod SMPL model
1. Download SMPL (https://smpl.is.tue.mpg.de/) model
2. copy `.pkl` model to `ROOT_DIR/smpl_model/*.pkl`
3. file structure:
```
- ROOT_DIR
  - smpl_model
    - basicmodel_m_lbs_10_207_0_v1.0.0.pkl
    - basicModel_f_lbs_10_207_0_v1.0.0.pkl
```

## Synthetic Data
### BuFF 
1. Download BuFF dataset (https://buff.is.tue.mpg.de/) and unzip it to 
```
DATA_DIR/BuFF/buff_release/sequences/subject_id/garment_action/garment_action_frame.ply
```
2. Download BuFF registration (If available) from CAPE (https://cape.is.tue.mpg.de/) and unzip it to
```
DATA_DIR/BuFF/buff_release/sequences/subject_id/garment_action/garment_action_frame.npz
```
3. Download gender and minimal body shape from CAPE (https://cape.is.tue.mpg.de/) and unzip it to
```
# The minimal body shape is unavailable yet; We are contacting authors of CAPE for releasing our fitted minimal body shape.
DATA_DIR/BuFF/buff_release/minimal_body_shape/subject_id/subject_id_param.pkl
DATA_DIR/BuFF/buff_release/minimal_body_shape/subject_id/subject_id_minimal.pkl
DATA_DIR/BuFF/buff_release/misc/smpl_tris.npy
DATA_DIR/BuFF/buff_release/misc/subj_genders.pkl
```
4. Render RGB image, rasterize depth image, save unprojected colorful point cloud
```
cd depth_renderer/
python buff_preprocessing.py
```

## Running
### Learn Fusion Shape via SDF
#### 1. Canonicalize Input Partial Shape

#### 2. Learn Canonical Fusion Shape

### 2. Learn Neural Surface Field based on Fusion Shape