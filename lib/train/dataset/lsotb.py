import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class LSOTB(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
       args:
           root - path to the lasot dataset.
           image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                           is used by default.
           vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                   videos with subscripts -1, -3, and -5 from each class will be used for training.
           split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                   vid_ids or split option can be used at a time.
           data_fraction - Fraction of dataset to be used. The complete dataset is used by default
       """
        root = env_settings().lasot_dir if root is None else root
        super().__init__('Lsotb', root, image_loader)

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = self.root
            if split == 'train':
                file_path = os.path.join(ltr_path, 'list.txt')
            else:
                raise ValueError('Unknown split name.')
            # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def get_name(self):
        return 'lasot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
       #class_seq = seq_name.split('-')[0]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        #valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        #visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox}

    def _get_frame_path(self, seq_path, frame_id):
        base_path = os.path.join(seq_path, '{:08}'.format(frame_id + 1))

        # 按优先级检查扩展名
        for ext in ['.jpg', '.jpeg']:
            file_path = base_path + ext
            if os.path.exists(file_path):
                return file_path

        # 如果都不存在，返回.jpg路径
        return base_path + '.jpg'

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-1]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):

        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta


if __name__ == '__main__':
    seq = LSOTB(root='/root/shared-nvme/datasets/data/train_data/LSOTB_TIR_train',split='train')
    res,__,_ = seq.get_frames(1,[1])
    print(res)