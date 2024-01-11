
## Dependencies
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
Our canonical pose is defined differently than in FITE. Thus please comment out https://github.com/jsnln/fite/blob/main/step1_diffused_skinning/compute_diffused_skinning.py#L54-55 to make sure `cpose` is all 0. <br>
Please copy the computed skinning weights field as well as the canonical pose mesh to `ROOT_DIR/diffused_smpl_skinning_field`. File structure should look like:
```
- ROOT_DIR
  - diffused_smpl_skinning_field
    - subject_cano_lbs_weights_grid_float32.npy
    - subject_minimal_cpose.ply
```
