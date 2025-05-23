from torch.utils.data import DataLoader
from .dataset_det3d import JRDBDet3D, NuScenesDet3D


def get_dataloader(split, batch_size, num_workers, shuffle, dataset_cfg):
    if dataset_cfg["name"] == "JRDB":
        # from .jrdb_detection_3d import JRDBDet3D

        ds = JRDBDet3D("data/JRDB2019", split, dataset_cfg)
    elif dataset_cfg["name"] == "JRDB22":
        # from .jrdb_detection_3d import JRDBDet3D

        ds = JRDBDet3D("data/JRDB2022", split, dataset_cfg)
    else:
        raise RuntimeError(f"Unknown dataset '{dataset_cfg['name']}'")

    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=ds.collate_batch,
    )
