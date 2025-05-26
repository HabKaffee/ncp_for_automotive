from src.model import Model, DrivingModelModule, CustomDataset
import argparse
import os
import torch
from torch import nn, optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import TensorBoardLogger

def train_model_with_parameters(epochs_to_train:int,
                                batch_size:int,
                                output_size:int,
                                units:int,
                                sequence_length:int,
                                # test_size:float,
                                lr:float,
                                device:str,
                                # data_path:str,
                                # annotations_file:str,
                                ):
    print(device)
    model_dir = f'model/max_ep_{epochs_to_train}_units_{units}_seq_{sequence_length}_lr_{lr}'
    os.makedirs(model_dir, exist_ok=True)
  
    datasets = [
        {'annotations_file': 'out/Town01_opt/data.txt', 'img_dir': 'out/Town01_opt/'},
        {'annotations_file': 'out/Town02_opt/data.txt', 'img_dir': 'out/Town02_opt/'},
        {'annotations_file': 'out/Town03_opt/data.txt', 'img_dir': 'out/Town03_opt/'},
        {'annotations_file': 'out/Town04_opt/data.txt', 'img_dir': 'out/Town04_opt/'},
    ]
    validation_ds_idx = 0 # Town01_opt

    dataset = CustomDataset(
        datasets=datasets,
        # annotations_file=annotations_file,
        # img_dir=data_path,
        sequence_length=sequence_length
    )

    # dataset.train_test_split(test_size=test_size, random_state=42)
    dataset.assign_train_val_by_dataset(val_dataset_index=validation_ds_idx)

    num_workers = max(os.cpu_count() - 2, 0)
    print(f"num_workers = {num_workers}")
    # Create dataloaders
    train_loader = DataLoader(
        Subset(dataset, dataset.train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        Subset(dataset, dataset.test_indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers
    )

    base_model = Model(output_size=output_size, units=units)

    # Initialize Lightning module
    lightning_model = DrivingModelModule(
        model=base_model,
        loss_func=nn.L1Loss(),
        optimizer_cls=optim.Adam,
        optimizer_kwargs={'lr': lr},
        stb_weights=[1.0, 0.0, 0.0]  # Only steering loss for now
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=model_dir,
        filename='epoch_{epoch:02d}_{val_loss:.5f}',
        save_top_k=-1,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',       # what metric to monitor
        patience=8,               # how many epochs to wait after no improvement
        mode='min',               # minimize the val_loss
        verbose=True              # log when triggered
    )

    logger = TensorBoardLogger("logs", name="driving_model")

    trainer = Trainer(
        max_epochs=epochs_to_train,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
        log_every_n_steps=10,
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        logger=logger,
        gradient_clip_val=5.0
    )
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def main():
    parser = argparse.ArgumentParser(description="Train a sequence-based steering model.")

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training.')
    parser.add_argument('--output_size', type=int, default=4,
                        help='Number of output neurons (e.g., 1 for steering angle).')
    parser.add_argument('--units', type=int, default=20,
                        help='Number of units in the LTC RNN layer.')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Default sequence length for RNN.')
    # parser.add_argument('--test_size', type=float, default=0.5,
    #                     help='Test data size (from 0 to 1).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu).')

    # parser.add_argument('--data_path', type=str, default='out/Town01_opt/',
    #                     help='Path to the folder containing images.')
    # parser.add_argument('--annotations_file', type=str, default='out/Town01_opt/data.txt',
    #                     help='Path to the CSV/txt file containing annotations.')

    # parser.add_argument('--distinct_folder', type=str, default='placeholder',
    #                     help="folder to distinct experiments")
    args = parser.parse_args()
    # if args.distinct_folder == 'placeholder':
    #     print('Provide folder name for experiment')
    #     exit(1)

    train_model_with_parameters(
        epochs_to_train = args.epochs,
        batch_size = args.batch_size,
        output_size = args.output_size,
        units = args.units,
        sequence_length = args.sequence_length,
        # test_size = args.test_size,
        lr=args.lr,
        device = args.device,
        # data_path = args.data_path,
        # annotations_file = args.annotations_file,
        # folder=args.distinct_folder
    )

if __name__ == "__main__":
    main()
