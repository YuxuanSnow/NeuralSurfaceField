
import numpy as np
import os

path = 'data_animation/aist_demo/seqs'

pose_list = []
transl_list = []

for file in sorted(os.listdir(path)):
    print(file)
    file_path = os.path.join(path, file)
    # Load data
    data = dict(np.load(file_path))
    pose_list.append(data['pose'])
    transl_list.append(data['transl'])

summed_file = {
    'pose': pose_list,
    'transl': transl_list
}

np.save('data_animation/aist_poses.npy', summed_file)

