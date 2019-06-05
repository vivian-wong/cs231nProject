from utils import * 

def load(data_dir, load_baseline_model=False):
    # Data augmentation and normalization for training
    # Just normalization for validation
    if load_baseline_model:
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224), # don't random crop 
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    # Load PyTorch dataloaders & visualize a few images in the dataset
    return load_data(data_dir, data_transforms)
    
def train(num_epochs= 25, transfer=True, load_baseline_model=False):
    
    # Set up for training 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if load_baseline_model:
        # load baseline
        model_ft = torch.load('./model_baseline.pt')
    else:
        model_ft = models.resnet152(pretrained=True)
    
    if transfer:
        # Freeze all layers except last layer
        for param in model_ft.parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    
    # CHANGE NUMBER OF FEATURES
    if load_baseline_model:
        model_ft.fc = nn.Linear(num_ftrs,4) 
    else:
        model_ft.fc = nn.Linear(num_ftrs,7) 

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs)
    return model_ft

def visualize(model_ft, load_baseline_model=False):
    print("Visualize some predictions")
    if load_baseline_model:
        visualize_model(model_ft, num_images = 18, rate_hazard=True) 
    else:
        visualize_model(model_ft)