import os
import csv
import yaml
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from torch.utils.data import DataLoader, Subset
from models.TempCNN_windowsdisturbance import TempCNN
from utils.Dataset10y_SW import MultiDisDataset as myDataset

# Configuration path
CONFIG_PATH = '/config/'

def load_config(config_name):
    """Load YAML configuration file."""
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config("config.yaml")
version = config['version']
pathout = f'/{config["output"]}/{config["name"]}/CAN_{version}_{config["lenghwind"]}y_test_{config["num_test"]}/'

# Create output directory if it doesn't exist
if not os.path.exists(pathout):
    os.makedirs(pathout)

# Save configuration to CSV
def save_config_to_csv(config, filepath):
    """Save configuration dictionary to a CSV file."""
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['key', 'value'])
        for key, value in config.items():
            writer.writerow([key, value])

save_config_to_csv(config, pathout + 'config.csv')

# Prepare CSV for epoch results
with open(pathout + 'epoch_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'loss', 'val_loss', 'train_acc', 'val_acc'])

# Extract configuration parameters
lenghwind = config["lenghwind"]
FEATURE_COLUMS = config["columns"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
cuda_device = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pytorch_accuracy(y_pred, y_true):
    """Compute the accuracy for a batch of predictions."""
    y_pred = y_pred.argmax(1)
    return (y_pred == y_true).float().mean() * 100

def pytorch_train_one_epoch(pytorch_network, optimizer, loss_function, sigma, scheduler):
    """Train the neural network for one epoch."""
    pytorch_network.train(True)
    if scheduler:
        scheduler.step()

    loss_sum, example_count = 0.0, 0
    for batch in train_loader:
        x, ytype, ydate, ydur = batch['sequence'].to(device), batch['labels']['label_type'].to(device), batch['labels']['label_date'].to(device), batch['labels']['label_dur'].to(device)

        optimizer.zero_grad()
        y_pred = pytorch_network(x)
        loss = sum(loss_function(y_pred[k], y) for k, y in [('type', ytype), ('date', ydate), ('dur', ydur)])
        loss.backward()
        optimizer.step()

        loss_sum += float(loss) * len(x)
        example_count += len(x)

    return loss_sum / example_count

def pytorch_test(pytorch_network, loader, loss_function, sigma):
    """Test the neural network on a DataLoader."""
    pytorch_network.eval()
    loss_sum, acc_sum, example_count = 0.0, 0.0, 0
    predstype, predsdate, predsdur = [], [], []
    truetype, truedate, truedur = [], [], []
    ids, xs, ys = [], [], []

    with torch.no_grad():
        for batch, id, coord_x, coord_y in loader:
            x = batch['sequence'].to(device)
            ytype, ydate, ydur = batch['labels']['label_type'].to(device), batch['labels']['label_date'].to(device), batch['labels']['label_dur'].to(device)

            y_pred = pytorch_network(x)
            y_pred_type, y_pred_date, y_pred_dur = y_pred['type'], y_pred['date'], y_pred['dur']

            predtype, preddate, preddur = torch.argmax(y_pred_type, 1), torch.argmax(y_pred_date, 1), torch.argmax(y_pred_dur, 1)

            # Reshape and collect predictions and true values
            for i in range(len(predtype)):
                predstype.append(predtype[i].item())
                predsdate.append(preddate[i].item())
                predsdur.append(preddur[i].item())
                truetype.append(ytype[i].item())
                truedate.append(ydate[i].item())
                truedur.append(ydur[i].item())
                ids.append(id[i].item())
                xs.append(coord_x[i].item())
                ys.append(coord_y[i].item())

            loss = sum(loss_function(y_pred[k], y) for k, y in [('type', ytype), ('date', ydate), ('dur', ydur)])
            loss_sum += float(loss) * len(x)
            acc_sum += pytorch_accuracy(y_pred['type'], ytype) * len(x)
            example_count += len(x)

    avg_loss = loss_sum / example_count
    avg_acc = acc_sum / example_count

    return avg_loss, predstype, predsdate, predsdur, truetype, truedate, truedur, avg_acc, ids, xs, ys

def pytorch_train(pytorch_network):
    """Train the neural network for a specified number of epochs."""
    pytorch_network.to(device)
    optimizer = torch.optim.Adam(pytorch_network.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    l_train_loss, l_valid_loss = [], []
    for epoch in range(1, num_epochs + 1):
        sigma = (num_epochs / 2) * math.exp(-epoch / 10)
        print(f'Epoch: {epoch}, LR: {scheduler.get_lr()}')

        train_loss = pytorch_train_one_epoch(pytorch_network, optimizer, loss_function, sigma, scheduler)
        valid_loss, _, _, _, _, _, _, avg_acc, _, _, _ = pytorch_test(pytorch_network, valid_loader, loss_function, sigma)

        print(f"Epoch {epoch}/{num_epochs}: loss: {train_loss}, val_loss: {valid_loss}, avg_acc: {avg_acc}")

        scheduler.step()
        l_train_loss.append(train_loss)
        l_valid_loss.append(valid_loss)

    test_loss, predstype, predsdate, predsdur, truetype, truedate, truedur, test_acc, ids, xs, ys = pytorch_test(pytorch_network, test_loader, loss_function, sigma)
    print(f"Test: Loss: {test_loss}, Accuracy: {test_acc}")

    id_result = np.hstack((np.stack(predstype), np.stack(predsdate), np.stack(predsdur), np.stack(truetype), np.stack(truedate), np.stack(truedur), np.stack(ids), np.stack(xs), np.stack(ys)))
    pd.DataFrame(id_result).to_csv(pathout + name + "_id_result.csv")

    fig, axes = plt.subplots(2, 1)
    axes[1].set_title('Train loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].plot(l_train_loss, label='Train')
    axes[1].plot(l_valid_loss, label='Validation')
    plt.tight_layout()
    plt.savefig(pathout + name + "_graph.png")
    plt.show()

if __name__ == '__main__':
    np.random.seed(5)
    random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Load validation and training data paths (containing one parquets for each tiles)
    pathValid = f'/Extraction/Parquet_{lenghwind}y_CANv{version}/valid/*.parquet'
    validfile = glob(pathValid, recursive=True)
    datafolder = f'/Extraction/Parquet_{lenghwind}y_CANv{version}/k/*.parquet'
    datafile = glob(datafolder, recursive=True)
    random.shuffle(datafile)

    n = 7  # 10kfold
    for i in range(0, len(datafile) - n + 1, n):
        foldnum = str(int(i / n))
        pathtest = datafile[i:i + n]
        pathTrain = list(set(datafile) - set(pathtest))

        testfile = [file for sublist in [glob(f, recursive=True) for f in pathtest] for file in sublist]
        trainfile = [file for sublist in [glob(f, recursive=True) for f in pathTrain] for file in sublist]

        train_dataset = myDataset(trainfile)
        test_dataset = myDataset(testfile)
        valid_dataset = myDataset(validfile)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        name = f'k_{foldnum}_TempCNN_CANv{version}'
        model = TempCNN(input_dim=len(FEATURE_COLUMS), n_type=config['nbclass'], n_date=lenghwind + 1, n_dur=lenghwind + 1, sequencelength=lenghwind)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params}")

        pytorch_train(model)
        end.record()
        torch.cuda.synchronize()

        print('Time, minutes', start.elapsed_time(end) / 60000)  # Convert milliseconds to minutes
        PATH = pathout + name + ".pt"
        torch.save(model, PATH)
