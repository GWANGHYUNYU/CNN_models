import os
import random
import pandas as pd
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import matplotlib.pyplot as plt

# CUDA Check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
# CUDA 기기가 존재한다면, 아래 코드가 CUDA 장치를 출력합니다:
print(device)

# parameters
file_name = 'Lenet5_STL10_1_GH.pt'
learning_rate = 0.001
training_epochs = 200
batch_size = 32
num_classes = 10

if not os.path.isdir('runs'):
    os.mkdir('runs')
# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
writer = SummaryWriter('./runs/Lenet5_STL10_Adam_0_001')


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])    # 3채널용 정규화
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])                   # Binary Image용 정규화

root = '/home/vips/share/Gwanghyun/data'
print(root)

trainset = datasets.STL10(root=root, split='train', download=True, transform=transform)
testset = datasets.STL10(root=root, split='test', download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')  # MNIST
classes = ('plane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')    #STL10

def show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
print(labels.size())

# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
# 이미지 보여주기
show(torchvision.utils.make_grid(images))

class LeNet_5(torch.nn.Module):
    def __init__(self, num_classes=num_classes, init_weights=False):
        super(LeNet_5,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)    # MNIST
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(256*12*12, 1024)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool3(x)

        x = x.view(-1, 256*12*12)
        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)
        x = self.relu2_fc2(x)
        
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model = LeNet_5().to(device)

# 모델 summary를 확인합니다.
summary(model, input_size=(3, 96, 96), device=device.type)   # MNIST

# loss function, optimizer 정의하기
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

best_valid_loss = float('inf')
best_loss = float('inf')

# 학습하기
for epoch in range(training_epochs):
    print("epoch: [%d/%d]" % (epoch+1, training_epochs+1))
    running_batch_loss = 0
    running_total_loss = 0
    model.train()

    correct = 0
    total = 0
    count = 0

    for i, data in enumerate(tqdm(trainloader)):
        images, labels = data[0].to(device), data[1].to(device)

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()
        loss.backward()

        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.step()

        # 오차값을 총 오차에 더함
        running_batch_loss += loss.item()
        running_total_loss += loss.item()

        # if i % 100 == 99:    # print every 1000 mini-batches
        #     # ...학습 중 손실(running loss)을 기록하고
        #     # writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
        #     print('[%d epoch, %5d mini_batchs] loss: %.10f' %(epoch + 1, i + 1, running_batch_loss / 100))
        #     running_batch_loss = 0.0

    print('[%d epoch, %5d iteration] Training loss: %.10f, Training acc: %4d' % (epoch + 1, i + 1, running_total_loss / len(trainloader), 100 * correct / total))
    
    writer.add_scalar('train_loss', (running_total_loss / len(trainloader)), epoch)
    writer.add_scalar('train_acc', (100 * correct / total), epoch)

    correct = 0
    total = 0
    count = 0
    
    model.eval()
    for i, data in enumerate(testloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        # inputs, labels = data
        images, labels = data[0].to(device), data[1].to(device)
    
        # 순전파 + 역전파 + 최적화를 한 후
        outputs = model(images)
        loss = criterion(outputs, labels)
    
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print("Validating LOSS: {}".format(loss.item()))
    print("Validating Accuracy of the model {} %".format(100 * correct / total))
    valid_loss = loss.item()

    writer.add_scalar('val_loss', np.mean(valid_loss), epoch)
    writer.add_scalar('val_acc', (100 * correct / total), epoch)

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './checkpoint/' + file_name)
        print('Model Saved!')
    
    # scheduler.step()
print('Finished Training')
writer.close()

# Load model and check accuracy
load_model = LeNet_5().to(device)
load_model.load_state_dict(torch.load('./checkpoint/' + file_name))

# Test model and check accuracy
correct = 0
total = 0

load_model.eval()
with torch.no_grad():

    for i, data in enumerate(testloader):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        # inputs, labels = data
        images, labels = data[0].to(device), data[1].to(device)

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = load_model(images)
        loss = criterion(outputs, labels)

        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("LOSS: {}".format(loss.item()))
print("Testing Accuracy of the model {} %".format(100 * correct / total))