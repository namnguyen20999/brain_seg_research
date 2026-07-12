import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    """
    Same as nnUNetTrainer but stops once the EMA pseudo Dice (the same metric nnU-Net
    already uses to pick checkpoint_best.pth) hasn't improved for `patience` epochs.

    Patience/min-epochs are read from env vars so they can be tuned per run without
    editing this file:
        NNUNET_ES_PATIENCE     (default 50)
        NNUNET_ES_MIN_EPOCHS   (default 100)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.early_stopping_patience = int(os.environ.get('NNUNET_ES_PATIENCE', 50))
        self.early_stopping_min_epochs = int(os.environ.get('NNUNET_ES_MIN_EPOCHS', 100))
        self._epochs_without_improvement = 0

    def on_epoch_end(self):
        prev_best = self._best_ema
        super().on_epoch_end()

        if prev_best is not None and self._best_ema is not None and self._best_ema <= prev_best:
            self._epochs_without_improvement += 1
        else:
            self._epochs_without_improvement = 0

        if (self.current_epoch >= self.early_stopping_min_epochs
                and self._epochs_without_improvement >= self.early_stopping_patience):
            self.print_to_log_file(
                f"Early stopping: no improvement in EMA pseudo Dice for "
                f"{self._epochs_without_improvement} epochs (patience={self.early_stopping_patience}). "
                f"Stopping at epoch {self.current_epoch}."
            )
            self.num_epochs = self.current_epoch

    def run_training(self):
        self.on_train_start()

        epoch = self.current_epoch
        while epoch < self.num_epochs:
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            epoch = self.current_epoch

        self.on_train_end()
