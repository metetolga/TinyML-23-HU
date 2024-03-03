import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import ToTensor, CustomDataset, save_model_and_stats, GaussianNoise, FlipSeg
from model import NeuralNet

from datetime import datetime

def main():
    EPOCH = args.epoch
    BATCH_SIZE = args.batch
    BATCH_SIZE_TEST = args.batch
    LR = args.lr
    SIZE = args.size
    data_path = args.data_path
    data_idx = args.data_idx
    device = torch.device("cuda") if args.cuda and torch.cuda.is_available() else torch.device("cpu")
    
    print(device)
    
    net = NeuralNet()
    net.to(device)

    augmentation = transforms.Compose([
        FlipSeg(),
        GaussianNoise()
    ])
    
    trainset = CustomDataset(
        root_dir=data_path,
        indice_dir=data_idx,
        mode='train',
        size=SIZE,
        transform=transforms.Compose([ToTensor()])
    )
    testset = CustomDataset(
        root_dir=data_path,
        indice_dir=data_idx,
        mode='test',
        size=SIZE,
        transform=transforms.Compose([ToTensor()])
    )

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
        
    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    print("Start training")
    
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for j, data in enumerate(trainloader):
            inputs, labels = data['data'], data['label']
            inputs = inputs.float()
            inputs = augmentation(inputs).to(device)
            # inputs = inputs.float().to(device)
            # print(inputs.shape)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            running_loss += loss.item()
            i += 1
        print('Epoch is %d \n Train Acc: %.5f Train loss: %.5f' % (epoch + 1, accuracy / j, running_loss / j))

        train_loss.append(running_loss / i)
        train_acc.append((accuracy / i).item())

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0

        for data_test in testloader:
            net.eval()
            IEGM_test, labels_test = data_test['data'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        test_loss.append(running_loss_test / i)
        test_acc.append((correct / total).item())

    stats = [train_loss, train_acc, test_loss, test_acc]
    save_model_and_stats(net, stats, model_name="flip_noise_50",  stats_save_name="stats_flip_noise_50")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cuda", help="use cuda", action="store_true")
    argparser.add_argument("--epoch", type=int,help="epoch number", default=1)
    argparser.add_argument("--lr", type=float, help="learning rate", default=0.0001)
    argparser.add_argument("--batch", type=int, help="batch size for data loader", default=32)
    argparser.add_argument("--size", type=int, help="data size per sample", default=1250)
    argparser.add_argument("--data_path", type=str, help="df: .\data_set\\", default='.\data_set\\')
    argparser.add_argument("--data_idx", type=str, help="df: .\data_indices", default='.\data_indices')
    args = argparser.parse_args()
    main()
