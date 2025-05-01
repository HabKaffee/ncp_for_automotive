from src.model import Model, Trainer

import argparse
import torch

def train_model_with_parameters(epochs_to_train:int,
                                batch_size:int,
                                output_size:int,
                                units:int,
                                sequence_length:int,
                                test_size:float,
                                lr:float,
                                device:str,
                                data_path:str,
                                annotations_file:str
                                ):
    print(device)

    model = Model(output_size, units)
    model.to(device)

    # loss_func = torch.nn.functional.mse_loss
    loss_func = torch.nn.functional.l1_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # + l2 reg 1e-5?

    trainer = Trainer(model,
                    loss_func,
                    optimizer,
                    annotations_file=annotations_file,
                    img_dir=data_path,
                    test_size=test_size,
                    random_state=42,
                    stb_weights=[1, 0, 0],
                    sequence_length=sequence_length,
                    batch_size=batch_size)
    trainer.train(epochs=epochs_to_train)
    model.save_model(f'model/lr_{lr}_sl_{sequence_length}_l1_{epochs_to_train}_epochs_{units}_units.pth')
    del model

def main():
    parser = argparse.ArgumentParser(description="Train a sequence-based steering model.")

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training.')
    parser.add_argument('--output_size', type=int, default=1,
                        help='Number of output neurons (e.g., 1 for steering angle).')
    parser.add_argument('--units', type=int, default=20,
                        help='Number of units in the LTC RNN layer.')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Default sequence length for RNN.')
    parser.add_argument('--test_size', type=float, default=0.5,
                        help='Test data size (from 0 to 1).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu).')

    parser.add_argument('--data_path', type=str, default='out/Town01_opt/',
                        help='Path to the folder containing images.')
    parser.add_argument('--annotations_file', type=str, default='out/Town01_opt/data.txt',
                        help='Path to the CSV/txt file containing annotations.')

    args = parser.parse_args()
    train_model_with_parameters(
        epochs_to_train = args.epochs,
        batch_size = args.batch_size,
        output_size = args.output_size,
        units = args.units,
        sequence_length = args.sequence_length,
        test_size = args.test_size,
        lr=args.lr,
        device = args.device,
        data_path = args.data_path,
        annotations_file = args.annotations_file,
    )

if __name__ == "__main__":
    main()
