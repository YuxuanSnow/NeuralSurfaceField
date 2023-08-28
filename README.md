# NSF: Neural Surface Field for Human Modeling from Monocular Depth
#### [Project Page](nsf.yuxuan-xue.com) | [Video](PlaceHolder) | [Paper](PlaceHolder)

In ICCV 2023, Paris

[Yuxuan Xue](https://yuxuan-xue.com/)<sup>1, *</sup>, [Bharat Lal Bhatnagar](https://virtualhumans.mpi-inf.mpg.de/people/Bhatnagar.html)<sup>2,*</sup>, [Riccardo Marin](https://ricma.netlify.app/)<sup>2</sup>, [Nikolaos Sarafianos](https://nsarafianos.github.io/)<sup>2</sup>, [Yuanlu Xu](https://web.cs.ucla.edu/~yuanluxu/)<sup>2</sup>, [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)<sup>1</sup>, [Tony Tung](https://sites.google.com/site/tony2ng/)<sup>2</sup>


<sup>1</sup>Real Virtual Human Group @ University of Tübingen & Tübingen AI Center & Max Planck Institute for Informatics \
<sup>2</sup>Meta Reality Lab Research 

## Background
### Install dependencies

Setup (with [conda](https://docs.conda.io/en/latest/)): 
with `environment.yml` file (tested on cuda 11.7):
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
pip install cython
```
In addition, install `kaolin` from source (https://kaolin.readthedocs.io/en/latest/notes/installation.html) and modify:
```
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.13.0
python setup.py develop
```
After installed `kaolin`, modify `trianglemesh.py` to enable different number of faces for multiple subject:
replace `KAOLIN_DIR/kaolin/kaolin/metrics/trianglemesh.py` with `ROOT_DIR/kaolin/trianglemesh.py`. Make sure that only function point_to_mesh_distance is modified


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

### Pre-diffuse SMPL skinning weights field
We use FITE (Lin et al., ECCV2022) to compute the smooth SMPL skinning weights field. 
Please refer to `https://github.com/jsnln/fite#3-diffused-skinning` for more information.
Please copy the computed skinning weights field as well as the canonical pose mesh to `ROOT_DIR/diffused_smpl_skinning_field`. File structure should look like:
```
- ROOT_DIR
  - diffused_smpl_skinning_field
    - subject_cano_lbs_weights_grid_float32.npy
    - subject_minimal_cpose.ply
```

## Running
### Learn Fusion Shape via SDF
#### 1. Canonicalize Input Partial Shape
canonicalize input partial shape by root finding, save to preprocessed file.
```
python buff_root_finding_male.py
python buff_root_finding_female.py
```

#### 2. Learn Canonical Fusion Shape
```
python buff_fusion_shape_male.py --exp_id 1 --batch_size 1 --split_file ./assets/data_split/buff_male_train_val.pkl --mode train --epochs 301
python buff_fusion_shape_female.py --exp_id 2 --batch_size 1 --split_file ./assets/data_split/buff_female_train_val.pkl --mode train --epochs 301
```

### 2. Learn Neural Surface Field based on Fusion Shape
```
python buff_nsf_male.py --exp_id 11 --batch_size 2 --split_file ./assets/data_split/buff_male_train_val.pkl --mode train --epochs 301
python buff_nsf_female.py --exp_id 12 --batch_size 2 --split_file ./assets/data_split/buff_female_train_val.pkl --mode train --epochs 301
```
Please adjust the hyperparameter for learning rate decay given the amount of samples in your training data.

The Marching Cube extracted fusion shapes have inverted normals. Handle it in `Meshlab` by: `Filters - Normals, Curvatures and Orientation - Invert face orientation` 
