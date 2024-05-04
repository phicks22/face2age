import time
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from image_dataset import ImageDataset, TestDataset
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Subset
from model import Model 
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
        action='store_true',
    )
    parser.add_argument(
        '-outfile',
        help='name of prediction dataframe .csv file',
        type=str,
        default='submission.csv',
    )
    args = parser.parse_args()   
    start = time.time()
    
    # Define variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    weights_file = Path(args.weights)

    if args.test_set:
        TEST_IMGS = DATA_DIR / 'wiki_labeled'
        labels_file = DATA_DIR / 'wiki_labels.csv'
        test_indices_file = DATA_DIR / "test_indices.txt"

        # Load data
        data = ImageDataset(
            labels_file, 
            TEST_IMGS,
            transform=ToPILImage()
        )
        
        test_indices = list(np.loadtxt(test_indices_file))
        
        # Define datasets for each 
        test_dataset = Subset(data, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    else:
        TEST_IMGS = DATA_DIR / 'wiki_judge_images'
        ids_file = DATA_DIR / 'wiki_judge.csv'
        
        # Load data
        data = TestDataset(
            ids_file, 
            TEST_IMGS,
            transform=ToPILImage()
        )
        
        # Define datasets for each 
        test_loader = DataLoader(data, batch_size=batch_size, num_workers=1, shuffle=False)

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

            z = data.to(device)
            pred_age = model(z.float())
            predictions['age'].append(pred_age.item())
            
            if args.test_set:
                predictions['ID'].append(_id[0].item())
            else:
                predictions['ID'].append(_id.item())
    # Save predictions
    pred_df = pd.DataFrame.from_dict(predictions)  
    outfile = RESULTS_DIR / args.outfile
    pred_df.to_csv(outfile, index=False)

