# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module() #decorator as a plugin registrar.
class Endovis2017RearrangedDataset(BaseSegDataset):
    """Endovis2017Dataset dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('background', 'Bipolar_forceps', 'Prograsp-Forceps', 
                 'Large-Needle-Driver', 'Vessel-Sealer', 'Grasping-Retractor',
                 'Monopolar-Curved-Scissors', 'Other'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [0, 255, 255], [255, 0, 255], [255, 255, 0], [0, 128, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, 
            seg_map_suffix=seg_map_suffix, 
            **kwargs)
