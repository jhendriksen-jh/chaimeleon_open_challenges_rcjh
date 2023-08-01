from library.datasets import ProstateCancerDataset
from library.models import ProstateImageModel, ProstateMetadataModel
from library.train import ImageTrainer, get_device, create_dataloader, create_optimizer, PROSTATE_LOSS

def main(data_directory: str, train: bool = False):
    if train:
        # import pudb; pudb.set_trace()
        train_dataset = ProstateCancerDataset(data_directory)
        val_dataset = ProstateCancerDataset(data_directory, split_type = 'val')

        image_model = ProstateImageModel()
        metadata_model = ProstateMetadataModel()

        device = get_device()
        train_loader = create_dataloader(train_dataset)
        val_loader = create_dataloader(val_dataset)
        optimizer = create_optimizer(image_model)

        trainer = ImageTrainer(image_model, train_loader, val_loader, PROSTATE_LOSS, optimizer, device)

        train_dataset[0]
        trainer.train(4)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--train', action='store_true', help = 'runs through training with given parameters')
    input_args = parser.parse_args()
    main(data_directory = input_args.data_dir, train = input_args.train)