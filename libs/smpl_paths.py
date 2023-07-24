
import os
import sys
import numpy as np
from os.path import join
import pickle as pkl
from libs.serialization import backwards_compatibility_replacements, load_model
from libs.geometry import get_hres
import scipy.sparse as sp

## Set your paths here
ROOT = '/mnt/qb/work/ponsmoll/yxue80/project/shapefusion/smpl_models'
smpl_vt_ft_path = join(ROOT, 'smpl_vt_ft.pkl')

class SmplPaths:
    def __init__(self, gender='neutral'):
        self.gender = gender

    def get_smpl_file(self):
        if self.gender == 'neutral':
            return join(ROOT,
                        'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')

        elif self.gender == 'male':
            return join(ROOT,
                             'basicmodel_m_lbs_10_207_0_v1.1.0.pkl')

        elif self.gender == 'female':
            return join(ROOT,
                             'basicmodel_f_lbs_10_207_0_v1.1.0.pkl')

        else:
            print('Please check the gender.')
            raise NotImplementedError

    def get_smpl(self):
        smpl_m = load_model(self.get_smpl_file())
        smpl_m.gender = self.gender
        return smpl_m

    def get_hres_smpl_model_data(self):

        dd = pkl.load(open(self.get_smpl_file()), encoding='latin-1')
        backwards_compatibility_replacements(dd)

        hv, hf, mapping = get_hres(dd['v_template'], dd['f'])

        num_betas = dd['shapedirs'].shape[-1]
        J_reg = dd['J_regressor'].asformat('csr')

        model = {
            'v_template': hv,
            'weights': np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]),
            'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207),
            'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
            'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
            'kintree_table': dd['kintree_table'],
            'bs_type': dd['bs_type'],
            'bs_style': dd['bs_style'],
            'J': dd['J'],
            'f': hf,
        }

        return model

    def get_hres_smpl(self):
        smpl_m = load_model(self.get_hres_smpl_model_data())
        smpl_m.gender = self.gender
        return smpl_m

    @staticmethod
    def get_vt_ft():
        vt, ft = pkl.load(open(smpl_vt_ft_path, 'rb'), encoding='latin-1')
        return vt, ft

    @staticmethod
    def get_vt_ft_hres():
        vt, ft = SmplPaths.get_vt_ft()
        vt, ft, _ = get_hres(np.hstack((vt, np.ones((vt.shape[0], 1)))), ft)
        return vt[:, :2], ft

    @staticmethod
    def get_template_file():
        fname = join(ROOT, 'template', 'template.obj')
        return fname

    @staticmethod
    def get_template():
        from psbody.mesh import Mesh
        return Mesh(filename=SmplPaths.get_template_file())

    @staticmethod
    def get_faces():
        fname = join(ROOT, 'template', 'faces.npy')
        return np.load(fname)

    @staticmethod
    def get_bmap():
        fname = join(ROOT, 'template', 'bmap.npy')
        return np.load(fname)

    @staticmethod
    def get_fmap():
        fname = join(ROOT, 'template', 'fmap.npy')
        return np.load(fname)

    @staticmethod
    def get_bmap_hres():
        fname = join(ROOT, 'template', 'bmap_hres.npy')
        return np.load(fname)

    @staticmethod
    def get_fmap_hres():
        fname = join(ROOT, 'template', 'fmap_hres.npy')
        return np.load(fname)

    @staticmethod
    def get_mesh(verts):
        from psbody.mesh import Mesh
        return Mesh(v=verts, f=SmplPaths.get_faces())


if __name__ == '__main__':
    dp = SmplPaths(gender='neutral')
    smpl_file = dp.get_smpl_file()

    print(smpl_file)
