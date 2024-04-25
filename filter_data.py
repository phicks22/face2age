import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    labels_file = Path('data/wiki_labels.csv')
    labels_df = pd.read_csv(labels_file)
    zero_indices = np.where(labels_df.age.to_numpy() == 0)[0]
    
    labels_df.drop(index=zero_indices, inplace=True)
    
    labels_df.to_csv(labels_file, index=False)

