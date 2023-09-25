## Render Depth Image from Scan
### BuFF 
1. Download BuFF dataset (https://buff.is.tue.mpg.de/) as well as textured scan from CAPE dataset (https://cape.is.tue.mpg.de/) and unzip it to 
```
DATA_DIR/BuFF/buff_release/sequences/subject_id/garment_action/garment_action_frame.ply
```
2. Download BuFF registration (If available) from CAPE (https://cape.is.tue.mpg.de/) and unzip it to
```
DATA_DIR/BuFF/buff_release/sequences/subject_id/garment_action/garment_action_frame.npz
```
3. Download gender and minimal body shape from CAPE (https://cape.is.tue.mpg.de/) and unzip it to
```
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
5. Get the train/split file
```
cd dataloaders
python data_split.py --gender male --mode train
python data_split.py --gender female --mode train
```
