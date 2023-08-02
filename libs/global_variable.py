# ROOT_DIR = '/mnt/qb/work/ponsmoll/yxue80/project/NeuralSurfaceField/'
# DATA_DIR = '/mnt/qb/work/ponsmoll/yxue80/project/NeuralSurfaceField/Data_scan/'
ROOT_DIR = '/mnt/lustre/ponsmoll/yxue80/project/NeuralSurfaceField/'
DATA_DIR = '/mnt/lustre/ponsmoll/yxue80/project/NeuralSurfaceField/Data_scan/'
# ROOT_DIR = '/home/yuxuan/project/NeuralSurfaceField/'
# DATA_DIR = '/home/yuxuan/project/NeuralSurfaceField/Data_scan/'

if ROOT_DIR.startswith('/home'):
    position = 9
elif ROOT_DIR.startswith('/mnt/lustre'):
    position = 11
else:
    position = 12