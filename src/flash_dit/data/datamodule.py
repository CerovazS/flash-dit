"""Lightning DataModule wrapping the pre-computed latent HDF5 dataset."""
import lightning as L
from torch.utils.data import DataLoader

from .latent_dataset import LatentDataset


class LatentDataModule(L.LightningDataModule):
    """DataModule for pre-computed audio latents stored in HDF5.

    Args:
        h5_path:    path to the HDF5 file produced by precompute_latents.py.
        batch_size: per-GPU batch size.
        num_workers: DataLoader worker count per process.
        pin_memory: pin tensors to CUDA-accessible memory.
    """

    def __init__(
        self,
        h5_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seq_len = seq_len

    def setup(self, stage: str | None = None) -> None:
        # Random crop only on train; val/test use the full stored latent for consistency.
        self.train_ds = LatentDataset(self.h5_path, split="train", seq_len=self.seq_len)
        self.val_ds   = LatentDataset(self.h5_path, split="val")
        self.test_ds  = LatentDataset(self.h5_path, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
