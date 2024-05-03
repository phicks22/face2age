import time
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torchvision.models as models
from image_dataset import ImageDataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Subset
from model import Model 
from argparse import ArgumentParser
from loss import LogCoshLoss, HuboshLoss


def assign_loss(loss: str):
    if loss == 'mse':
        return nn.MSELoss()
    elif loss == 'log_cosh':
        return LogCoshLoss()
    elif loss == 'huber':
        return nn.SmoothL1Loss(delta=0.1)
    elif loss == 'hubosh':
        return HuboshLoss()
    else:
        raise ValueError('Choose a valid loss function')

def adjust_learning_rate(lr_start: float, epoch: int) -> float:
    lr = lr_start * (0.01 ** (epoch // 30))
    return lr


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-loss',
        help='Which loss function to use',
        type=str,
        choices=['mse', 'log_cosh', 'huber', 'hubosh'],
        default='mse',
    )
    parser.add_argument(
        '-epochs',
        help='number of epochs to run',
        type=int,
        default=25,
    )
    parser.add_argument(
        '--save',
        help='if applied, will save the model weights',
        action='store_true',
    )
    parser.add_argument(
        '-batch_size',
        help='size of each batch',
        type=int,
        default=128,
    )
    parser.add_argument(
        '-lr',
        help='learning rate',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '-train_size',
        help='percent of data used for training',
        type=float,
        default=0.7,
    )
    parser.add_argument(
        '-val_size',
        help='percent of data used for validation',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '-test_size',
        help='percent of data used for testing',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '-outdir',
        help='directory to save results to',
        type=str,
        default='results',
    )
    parser.add_argument(
        '-out_prefix',
        help='prefix of results outfiles',
        type=str,
        default=None,
    )
    args = parser.parse_args()   
    start = time.time()    

    # Set paths
    DATA_DIR = Path('data')
    TRAIN_IMGS = DATA_DIR / 'wiki_labeled'
    TEST_IMGS = DATA_DIR / 'wiki_judge_images'

    labels_file = DATA_DIR / 'wiki_labels.csv'
    judge_ids_file = DATA_DIR / 'wiki_judge.csv'

    # Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_size = args.train_size
    validation_size = args.val_size
    test_size = args.test_size
    batch_size = args.batch_size
    epochs = args.epochs
    save_model = args.save
    loss_fn = assign_loss(args.loss)
    lr = args.lr

    # Get training data
    train_data = ImageDataset(
        labels_file, 
        TRAIN_IMGS,
        transform=ToPILImage()
    )

    # Get indices for train-validation-test split
    all_indices = list(range(len(train_data)))
    train_indices, test_indices = train_test_split(
        all_indices, 
        test_size=test_size, 
        random_state=42,
    )
    train_indices, val_indices = train_test_split(
        train_indices, 
        test_size=validation_size/(1-test_size), 
        random_state=42,
    )

    # Save test indices
    test_indices_file = DATA_DIR / "test_indices.txt"
    with open(test_indices_file, 'w') as f:
        for index in test_indices:
            f.write(str(index) + '\n')

    # Define datasets for each 
    train_dataset = Subset(train_data, train_indices)
    val_dataset = Subset(train_data, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    # Initialize model
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    print('Using device:', device)
    training_metrics = {'epoch': [], 'loss': []}
    val_metrics = {'epoch': [], 'loss': []}
    for epoch in range(1, epochs+1):
        run_result = {'nsamples': 0, 'loss': 0}
        
        #alr = adjust_learning_rate(0.0005, epoch)
        #optimizer = optim.Adam(model.parameters(), lr = alr)
        for p in model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()
        
        train_bar = tqdm(train_loader)
        for data, target in train_bar:
            batch_size = data.size(0)
            run_result['nsamples'] += batch_size
            
            label = target.to(device).unsqueeze(1).float()
            z = data.to(device)
            pred_age = model(z.float())

            model.zero_grad()
            loss = loss_fn(pred_age, label)
            loss.backward()
            optimizer.step()
            
            run_result['loss'] += loss.item() * batch_size
           
        training_metrics['loss'].append(run_result['loss'] / run_result['nsamples'])
        training_metrics['epoch'].append(epoch)
        model.eval()

        # Test on validation set
        batch_loss = []
        valid_result = {'nsamples': 0, 'loss': 0}
        with torch.no_grad():
            valid_bar = tqdm(val_loader)
            for data, target in valid_bar:
                batch_size = data.size(0)
                valid_result['nsamples'] += batch_size

                label = target.to(device).unsqueeze(1).float() 
                z = data.to(device)
                pred_age = model(z.float())
                
                loss = loss_fn(pred_age, label)
                valid_result['loss'] += loss.item() * batch_size

            val_metrics['loss'].append(valid_result['loss'] / valid_result['nsamples'])
            val_metrics['epoch'].append(epoch)            

    # Save model 
    outdir = Path(args.outdir)
    if save_model:
        print('Saving Model.')
        torch.save(model.state_dict(), outdir / f'pretrained_weights/model_epochs-{epochs}.pytorch')

    # Save training and validation metrics
    print('Saving training/validation metrics.')
    if args.out_prefix is not None:
        train_outfile = f'{args.out_prefix}_training_loss.csv'
        val_outfile = f'{args.out_prefix}_val_loss.csv'
    else:
        train_outfile = f'{args.loss}_training_loss.csv'
        val_outfile = f'{args.loss}_val_loss.csv'

    training_metrics_df = pd.DataFrame.from_dict(training_metrics)
    val_metrics_df = pd.DataFrame.from_dict(val_metrics)

    training_metrics_df.to_csv(outdir / train_outfile, index=False)
    val_metrics_df.to_csv(outdir / val_outfile, index=False)
    
    print(f'Model trained in {(time.time() - start) / 60:.1f} mins.')
