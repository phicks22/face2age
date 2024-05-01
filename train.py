import torch
import torch.nn
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torchvision.models as models
from image_dataset import ImageDataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Subset


# Set paths
DATA_DIR = Path('data')
TRAIN_IMGS = DATA_DIR / 'wiki_labeled'
TEST_IMGS = DATA_DIR / 'wiki_judge_images'

labels_file = DATA_DIR / 'wiki_labels.csv'
judge_ids_file = DATA_DIR / 'wiki_judge.csv'

# Define network
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        # Load pre-trained VGG-16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze parameters
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        out = self.vgg16(x)
        
        return out
    

class AgePredictor(nn.Module):
    def __init__(self):
        super(AgePredictor, self).__init__()
        
        self.flat = nn.Flatten()
        self.l0 = nn.Linear(1000, 300)
        self.l1 = nn.Linear(300, 100)
        self.l2 = nn.Linear(100, 1)
        
        self.body = nn.Sequential(
            self.flat,
            self.l0,
            nn.ReLU(inplace=True),
            self.l1,
            nn.ReLU(inplace=True),
            self.l2,
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.body(x)
        return out
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.vgg = VGG16FeatureExtractor()
        self.age_predictor = AgePredictor()
        
    def forward(self, x):
        vgg_out = self.vgg(x)
        age_out = self.age_predictor(vgg_out)
        
        return age_out


# Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = 0.7
validation_size = 0.2
test_size = 0.1
batch_size = 128
epochs = 25
save_model = True

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

# Define datasets for each partition
train_dataset = Subset(train_data, train_indices)
val_dataset = Subset(train_data, val_indices)
test_dataset = Subset(train_data, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

loss = 'MSE'
if loss == 'MSE':
    loss_fn = nn.MSELoss()
model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)


print('Using device:', device)
training_metrics = {'epoch': [], 'loss': []}
for epoch in range(1, epochs+1):
    run_result = {'nsamples': 0, 'loss': 0}
    
    for p in model.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()
    
    train_bar = tqdm(train_loader)
    for data, target in train_bar:
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size
        
        label = target.to(device)
        z = data.to(device)
        pred_age = model(z.float())

        ######### Train generator #########
        label = label.unsqueeze(1)
        label = label.float()
        model.zero_grad()
        loss = loss_fn(pred_age, label)
        loss.backward()
        optimizer.step()

training_metrics_df = pd.DataFrame.from_dict(training_metrics)
training_metrics_df.to_csv(f'results/{loss}_training_loss.csv')
