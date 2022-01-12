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
file_name = 'AlexNet_STL10_1_GH.pt'
learning_rate = 0.001
training_epochs = 100
batch_size = 32
num_classes = 10

if not os.path.isdir('runs'):
    os.mkdir('runs')
# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
writer = SummaryWriter('./runs/AlexNet_STL10_1')

transform = transforms.Compose([transforms.ToTensor()])

root = '/home/vips/share/Gwanghyun/data'
print(root)

trainset = datasets.STL10(root=root, split='train', download=True, transform=transform)
testset = datasets.STL10(root=root, split='test', download=True, transform=transform)

# calculate the mean and standard deviation of trainset
meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in trainset]
stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in trainset]

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(meanR, meanG, meanB)
print(stdR, stdG, stdB)

# in paper, using normalize, horizontal reflection
# transform을 적용하다가 "TypeError: pic should be Tensor or ndarray. Got <class 'PIL.Image.Image'>." 에러가 날 경우 아래와 같은 순서로 입력해야함.
# train_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])])

test_transform = transforms.Compose([
                transforms.Resize(227),
                transforms.ToTensor(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])])

# apply transformation to train_ds and test0_ds
trainset.transform = train_transform
testset.transform = test_transform

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

STL10_classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
CIFAR10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
print(labels.size())

# 정답(label) 출력
print(' '.join('%5s' % STL10_classes[labels[j]] for j in range(batch_size)))
# 이미지 보여주기
# show(torchvision.utils.make_grid(images))

# Model
class AlexNet(nn.Module):
    def __init__(self, num_classes=num_classes, init_weights=False):
        super(AlexNet, self).__init__()
        # input size : (b x 3 x 227 x 227)
        # 논문에는 image 크기가 224 pixel이라고 나와 있지만, 오타입니다.
        # 227x227을 사용합니다.

        # Conv layer
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  # (b x 96 x 55 x 55)
            nn.ReLU(),  # inplace=True 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
                        # inplace=False in nn.ReLU and nn.LeakyReLU 해야함
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),   # original code는 위와 같지만 본 데이터셋에서는 작동을 안해서 스킵함
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 96 x 27 x 27)    # 2 x 2로 변환

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),   # original code는 위와 같지만 본 데이터셋에서는 작동을 안해서 스킵함
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 256 x 13 x 13)   # 2 x 2로 변환

            nn.Conv2d(256, 384, 3, 1, 1),  # (b x 384 x 13 x 13)
            nn.ReLU(),

            nn.Conv2d(384, 384, 3, 1, 1),  # (b x 384 x 13 x 13)
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, 1, 1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (b x 256 x 6 x 6)    # 2 x 2로 변환
        )

        # fc layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        # weight initialization
        if init_weights:
            self._initialize_weights()

    # define weight initialization function
    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # in paper, initialize bias to 1 for conv2, 4, 5 layer
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256*6*6)
        x = self.classifier(x)
        return x

model = AlexNet().to(device)

# 모델 summary를 확인합니다.
summary(model, input_size=(3, 227, 227), device=device.type)

# check weight initialization
# for p in model.parameters():
#     print(p)
#     break

# loss function, optimizer 정의하기
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

best_valid_loss = float('inf')
best_valid_acc = 0

# 학습하기
for epoch in range(training_epochs):
    print("epoch: [%d/%d]" % (epoch+1, training_epochs+1))
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
        running_total_loss += loss.item()

    print('[%d epoch, %5d iteration] Training loss: %.10f, Training acc: %4f%' % (epoch + 1, i + 1, running_total_loss / len(trainloader), 100 * correct / total))
    
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
    # valid loss 기반 모델 저장
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './checkpoint/' + file_name)
        print('Model Saved!')
    # valid acc 기반 모델 저장    
    if correct/total > best_valid_acc:
        best_valid_loss = correct/total
        torch.save(model.state_dict(), './checkpoint/' + file_name)
        print('Model Saved!')
    
    # scheduler.step()
print('Finished Training')
writer.close()

# Load model and check accuracy
load_model = AlexNet().to(device)
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