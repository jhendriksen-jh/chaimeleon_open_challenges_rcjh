import torch
from library.datasets import ProstateCancerDataset
from library.models import ProstateImageModel, ProstateMetadataModel, ProstateCombinedModel, get_number_of_parameters
from library.train import Trainer, get_device, create_dataloader, create_optimizer, PROSTATE_LOSS

def main(data_directory: str, train: bool = False):
    if train:
        # import pudb; pudb.set_trace()
        train_dataset = ProstateCancerDataset(data_directory)
        val_dataset = ProstateCancerDataset(data_directory, split_type = 'val')

        image_model = ProstateImageModel()
        metadata_model = ProstateMetadataModel()

        device = get_device()
        train_loader = create_dataloader(train_dataset, batch_size=256)
        val_loader = create_dataloader(val_dataset, batch_size=256)
        optimizer = create_optimizer(image_model)

        num_epochs = 50

        print(f"\n######## Training Image Model - {get_number_of_parameters(image_model)} params ########\n")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor = 0.5, patience = 4, verbose=True)
        ProstateImageTrainer = Trainer(image_model, train_loader, val_loader, PROSTATE_LOSS, optimizer, device, scheduler=scheduler)
        ProstateImageTrainer.train(num_epochs)
        print(f"Training averaged {sum(ProstateImageTrainer.train_time)/num_epochs:.2f}s per epoch")
        # ProstateImageTrainer.plot_acc()
        # ProstateImageTrainer.plot_loss()

        print(f"\n######## Training Metadata Model - {get_number_of_parameters(metadata_model)} ########\n")

        metadata_optimizer = create_optimizer(metadata_model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(metadata_optimizer, mode='max', factor = 0.5, patience = 4, verbose=True)
        ProstateMetadataTrainer = Trainer(metadata_model, train_loader, val_loader, PROSTATE_LOSS, metadata_optimizer, device, scheduler=scheduler)
        ProstateMetadataTrainer.train(num_epochs)
        print(f"Training averaged {sum(ProstateMetadataTrainer.train_time)/num_epochs:.2f}s per epoch")
        # ProstateMetadataTrainer.plot_acc()
        # ProstateMetadataTrainer.plot_loss()

        combo_model = ProstateCombinedModel()
        print(f"\n######## Training Combined Model - {get_number_of_parameters(combo_model)} ########\n")

        combo_optimizer = create_optimizer(combo_model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(combo_optimizer, mode='max', factor = 0.5, patience = 4, verbose=True)
        ProstateCombinedTrainer = Trainer(combo_model, train_loader, val_loader, PROSTATE_LOSS, combo_optimizer, device, scheduler=scheduler)
        ProstateCombinedTrainer.train(num_epochs)
        print(f"Training averaged {sum(ProstateCombinedTrainer.train_time)/num_epochs:.2f}s per epoch")
        # ProstateCombinedTrainer.plot_acc()
        # ProstateCombinedTrainer.plot_loss()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--train', action='store_true', help = 'runs through training with given parameters')
    input_args = parser.parse_args()
    main(data_directory = input_args.data_dir, train = input_args.train)