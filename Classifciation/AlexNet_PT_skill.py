import os
import random
import pandas as pd
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
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
file_name = 'AlexNet_PT_STL10_2_GH.pt'
learning_rate = 0.001
training_epochs = 10
batch_size = 32
num_classes = 10

if not os.path.isdir('runs'):
    os.mkdir('runs')
# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
writer = SummaryWriter('./runs/AlexNet_PT_STL10_2')

root = '/home/vips/share/Gwanghyun/data'
print(root)

# in paper, using normalize, horizontal reflection
# transform을 적용하다가 "TypeError: pic should be Tensor or ndarray. Got <class 'PIL.Image.Image'>." 에러가 날 경우 아래와 같은 순서로 입력해야함.
# train_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])

test_transform = transforms.Compose([
                transforms.Resize(227),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainset = datasets.STL10(root=root, split='train', download=True, transform=train_transform)
testset = datasets.STL10(root=root, split='test', download=True, transform=test_transform)
print(trainset.data.shape)
print(trainset.labels.shape)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

STL10_classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
CIFAR10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
print(labels.size())

# 이미지 그리드를 만듭니다.
img_grid = torchvision.utils.make_grid(images)

# 이미지를 보여줍니다.
matplotlib_imshow(img_grid, one_channel=True)

# tensorboard에 기록합니다.
writer.add_image(str(batch_size) +'_STL10_images', img_grid)


# 모델 정의하기
AlexNet_pt_model = models.alexnet(pretrained=True)                               # Pretrained model 설정
last_nn_in_features = AlexNet_pt_model.classifier[6].in_features                 # Pretrained model의 last layer의 출력 노드의 갯수
AlexNet_pt_model.classifier[6] = nn.Linear(last_nn_in_features, num_classes)     # Pretrained model의 last layer에 FC를 추가

model = AlexNet_pt_model.to(device)

# 모델 summary를 확인합니다.
summary(model, input_size=(3, 227, 227), device=device.type)

writer.add_graph(model, images.to(device))

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

    print('[%d epoch, %5d iteration] Training loss: %.10f, Training acc: %3d' % (epoch + 1, i + 1, running_total_loss / len(trainloader), 100 * correct / total))
    
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
    valid_acc = 100 * correct / total

    writer.add_scalar('val_loss', np.mean(valid_loss), epoch)
    writer.add_scalar('val_acc', valid_acc, epoch)

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    # # valid loss 기반 모델 저장
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     torch.save(model.state_dict(), './checkpoint/' + file_name)
    #     print('Model Saved!')

    # valid acc 기반 모델 저장
    # print('best_valid_acc', best_valid_acc)
    # print('valid_acc', valid_acc)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), './checkpoint/' + file_name)
        print('Model Saved!')
    
    # scheduler.step()
print('Finished Training')
writer.close()

# Load model and check accuracy
load_model = model
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