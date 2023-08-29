### Learn implicit Fusion Shape from input partial shape

##### 1. Canonicalize Input Partial Shape
canonicalize input partial shape by root finding, save to preprocessed file.
```
python buff_root_finding_male.py
python buff_root_finding_female.py
```

##### 2. Learn Canonical Fusion Shape
```
python buff_fusion_shape_male.py --exp_id 1 --batch_size 1 --split_file ./assets/data_split/buff_male_train_val.pkl --mode train --epochs 301
python buff_fusion_shape_female.py --exp_id 2 --batch_size 1 --split_file ./assets/data_split/buff_female_train_val.pkl --mode train --epochs 301
```

##### 3. Obtain Canonical Fusion Shape (Inference)
```
# Run Marching Cube on SDF
python buff_fusion_shape_male.py --exp_id 1 --batch_size 1 --split_file ./assets/data_split/buff_male_train_val.pkl --mode fusion_shape --save_name Recon_256
python buff_fusion_shape_female.py --exp_id 2 --batch_size 1 --split_file ./assets/data_split/buff_female_train_val.pkl --mode fusion_shape --save_name Recon_256

# Fit SMPL to SDF
python buff_fusion_shape_male.py --exp_id 1 --batch_size 1 --split_file ./assets/data_split/buff_male_train_val.pkl --mode fit_smpl
python buff_fusion_shape_female.py --exp_id 2 --batch_size 1 --split_file ./assets/data_split/buff_female_train_val.pkl --mode fit_smpl

# project off-surface points to fusion shape
python buff_fusion_shape_male.py --exp_id 1 --batch_size 1 --split_file ./assets/data_split/buff_male_train_val.pkl --mode projection
python buff_fusion_shape_female.py --exp_id 2 --batch_size 1 --split_file ./assets/data_split/buff_female_train_val.pkl --mode projection
```
The Marching Cube extracted fusion shapes have inverted normals. Handle it in `Meshlab` by: `Filters - Normals, Curvatures and Orientation - Invert face orientation` 
