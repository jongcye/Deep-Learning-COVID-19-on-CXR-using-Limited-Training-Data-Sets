# Import modules
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
from classification import header
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.utils import initialize_model
from utils.utils import plot_confusion_matrix
from utils.utils import most_common_top_1
from utils.customloader import COVID_Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

## Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load data
data_dir = header.data_dir

# Model name
model_name = header.model

# Number of classes
num_classes = header.num_classes

# Feature extract
feature_extract = header.feature_extract

# Test epoch
test_epoch = header.inference_epoch

def main():
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    print("Initializing Datasets and Dataloaders...")

    # Create training and test datasets
    test_dataset  = COVID_Dataset((header.img_size, header.img_size), n_channels=3, n_classes=4, mode='test')

    image_datasets = {'test': test_dataset}

    batch_size = {'test': header.test_batch_size}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size[x], num_workers=4, pin_memory=True) for x in ['test']}

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    since = time.time()

    # Load best model
    model = model_ft
    checkpoint = torch.load(os.path.join(header.save_dir, str(header.inference_epoch) + '.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    repeat = header.repeat

    # Each epoch has a training and validation phase
    for phase in ['test']:

        y_pred_total = []
        for x in range(445):
            y_pred_total.append([])

        for i in range(repeat):

            if phase == 'test':
                model.eval()  # Set model to evaluation mode
            else:
                print('Error: Phase should be set to test.')

            running_corrects = 0

            y_true = []
            y_pred = []
            y_prob = []

            idx = 1

            # Iterate over data.
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device=device, dtype=torch.long)

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                # statistics
                running_loss = loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                prob = torch.sigmoid(outputs)
                prob_np = prob.detach().cpu().numpy()
                labels_np = np.asarray(labels.cpu())
                pred_np = np.asarray(preds.cpu())

                for x in range(len(labels)):
                    y_pb = prob_np[x]
                    y_prob.append(y_pb)

                for x in range(len(labels)):
                    y_tr = labels_np[x]
                    y_true.append(y_tr)

                for x in range(len(inputs)):
                    y_pr = pred_np[x]
                    y_pred.append(y_pr)

                idx += 1

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)
            epoch_f1 = f1_score(y_true, y_pred, average='macro')

            print('{} Number: {:.1f} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, i, epoch_loss, epoch_acc, epoch_f1))

            for idx, item in enumerate(y_pred):
                y_pred_total[idx].append(item)

    y_pred = []

    for x in range(445):
        final_predict = most_common_top_1(y_pred_total[x])
        y_pred.append(final_predict)

    print()

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=['Normal', 'Bacteria', 'TB', 'Virus_or_COVID-19'], normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues)
    plt.show()

    # Overall classification report
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Bacteria', 'TB', 'Virus_or_COVID-19']))
    ACC = accuracy_score(y_true, y_pred)
    PREC = precision_score(y_true, y_pred, average='macro')
    REC = recall_score(y_true, y_pred, average='macro')
    F1 = f1_score(y_true, y_pred, average='macro')

    print('Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(ACC, PREC, REC, F1))

    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__=='__main__':
    main()