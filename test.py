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
from network import Model 
from argparse import ArgumentParser
from loss import RMSLE


DATA_DIR = Path('data')
RESULTS_DIR = Path('results')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-batch_size',
        help='size of each batch',
        type=int,
        default=128,
    )
    parser.add_argument(
        '-weights',
        help='/path/to/model_weights.pytorch',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--test_set',
    )
    parser.add_argument(
        '-outfile',
        help='name of prediction dataframe .csv file',
        type=str,
        default='submission.csv',
    )
    args = parser.parse_args()   
    start = time.time()

    if args.test_set:
        TEST_IMGS = DATA_DIR / 'wiki_labeled'
        ids_file = DATA_DIR / 'wiki_labels.csv'

    else:
        TEST_IMGS = DATA_DIR / 'wiki_judge_images'
        ids_file = DATA_DIR / 'wiki_judge.csv'
    
    # Define variables
    batch_size = args.batch_size
    weights_file = Path(args.weights)
    
    # Define model
    weights = torch.load(weights_file)
    model = Model()
    model.to(device)
    model.load_state_dict(weights) 
    model.eval()   

    predictions = {'ID': [], 'age': []} 
    with torch.no_grad():
        predict_bar = tqdm(test_loader)
        for data, _id in predict_bar:
            batch_size = data.size(0)
            valid_result['nsamples'] = batch_size

            z = data.to(device)
            pred_age = model(z.float())
            predictions['age'].append(pred_age.to('cpu').numpy()[0][0])
            predictions['ID'].append(_id[0])
    
    # Save predictions
    pred_df = pd.DataFrame.from_dict(predictions)  
    outfile = RESULTS_DIR / args.outfile
    pred_df.to_csv(outfile, index=False)

