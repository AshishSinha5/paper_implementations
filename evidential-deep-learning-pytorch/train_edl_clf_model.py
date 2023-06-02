import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader


from sklearn.metrics import accuracy_score

from edl_clf_model import edl_clf_model
from data_loader import MNIST, EDL_MNISTLoader


batch_size = 8
shuffle = True
epochs = 40
initial_lr = 0.001
units = [28*28, 30]

print(torch.__version__) # Get PyTorch and CUDA version
print(f"{torch.cuda.is_available() = }") # Check that CUDA works
print(f"{torch.cuda.device_count() = }") # Check how many CUDA capable devices you have
# Print device human readable names
print(f"{torch.cuda.get_device_name(0) = }")
# Add more lines with +1 like get_device_name(3), get_device_name(4) if you have more devices.

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.set_device(0)
print(f'Device = {device}')

def main():
    data_path = '../data/mnist/raw/mnist_train.csv'
    test_data_path = '../data/mnist/raw/mnist_test.csv'

    data = MNIST(data_path)
    test_data = MNIST(test_data_path)

    test_inputs = test_data.inputs
    test_labels = test_data.labels

    dataset = EDL_MNISTLoader(data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    clf = edl_clf_model().to(device)
    optimizer = optim.Adam(clf.parameters(), lr=initial_lr)

    results = []
    start_time = datetime.datetime.now()
    for epoch in range(epochs):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(dataloader))
        running_loss = 0
        
        for i, batches in tqdm(enumerate(dataloader)):
            input = batches[0].to(device)
            label = batches[1].to(device)

            optimizer.zero_grad()

            pred, alpha = clf.forward(input)
            loss = clf.loss(alpha, label)
            loss.backward()
            optimizer.step()

            scheduler.step()
            running_loss += loss.item()

            if (i > 0) and (i%400 == 0):
                with torch.no_grad():
                    train_labels = np.argmax(label.detach().cpu().numpy(), axis = 1)
                    train_pred = np.argmax(pred.detach().cpu().numpy(), axis = 1)
                    pred, alpha = clf.forward(torch.FloatTensor(test_inputs).to(device))
                    pred = pred.detach().cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    train_accuracy = accuracy_score(train_labels, train_pred)
                    accuracy = accuracy_score(test_labels, pred)
                    # roc_auc = roc_auc_score(test_labels, pred)
                print("Epoch: {}, Seq: {:,}/{:,}, " \
                    "Loss: {:.4f}, VAL_ACCURACY : {:.4f}, Lr: {:.6f}, TRAIN_ACCURACY : {:.4F}".format(epoch, i, len(dataloader), running_loss,
                                                                            accuracy, optimizer.param_groups[0]['lr'], 
                                                                            train_accuracy))
                results.append([epoch, i, running_loss, accuracy])
                running_loss = 0

        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
        state_dict_path = f'../models/edl/edl_clf_model_epoch_{epoch}_{current_datetime}_acc_{accuracy}.pt'
        torch.save(clf.state_dict(), state_dict_path)
        print(f'Model state dict saved to {state_dict_path}')


    end_time = datetime.datetime.now()  
    time_diff = round((end_time - start_time).total_seconds() / 60, 2)
    print('Total time taken: {:,} minutes'.format(time_diff))

    # Save results
    results_df = pd.DataFrame(results, columns=['epoch', 'batches', 'loss', 'accuracy'])
    results_df.to_csv('../model_metrics/edl/edl_clf_model_metrics.csv', index=False)

if __name__ == "__main__":
    main()