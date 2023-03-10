from __future__ import print_function, absolute_import

import os
import os.path as osp
import re

import numpy as np
import glob

# from utils import mkdir_if_missing, write_json, read_json

from IPython import embed


class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501' # 数据文件路径

    def __init__(self, root = 'data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # print(self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # data_dir ID CAMID NUM
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    # 检查路径是否存在
    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not avaliable".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not avaliable".format((self.train_dir)))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel = False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1 : continue
            pid_container.add(pid)
        pid2label = {pid : label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)
        num_imgs = len(img_paths)
        return dataset, num_pids, num_imgs

"""Create dataset"""

__img_factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    # 'dukemtmcreid': DukeMTMCreID,
    # 'msmt17': MSMT17,
}

__vid_factory = {
#     'mars': Mars,
#     'ilidsvid': iLIDSVID,
#     'prid': PRID,
#     'dukemtmcvidreid': DukeMTMCVidReID,
}

def get_names():
    return list(__img_factory.keys()) + list(__vid_factory.keys())

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

def init_vid_dataset(name, **kwargs):
    if name not in __vid_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __vid_factory.keys()))
    return __vid_factory[name](**kwargs)

if __name__ == '__main__':
    data = Market1501('/home/prj/data')