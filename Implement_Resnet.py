import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import gc


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
        # mean or std  can be calculated manually, but are also available online.
    )

    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
            root= data_dir, train=False,
            download=False, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset , data_dir : Download data here
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=False, transform=transform,
    ) #
    # http://www.cs.toronto.edu/~kriz/cifar.html

    # do not need this operation
    # valid_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=transform,
    # )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx) # 保证样本不重复
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    # just index not same
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


# CIFAR10 dataset
train_loader, valid_loader = data_loader(data_dir='./data',
                                         batch_size=8)
test_loader = data_loader(data_dir='./data',
                          batch_size=8,
                          test=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
            # In general, the number of channels * 2, the size becomes 1/2
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, Resblock, Resblocknumlist, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64 #  embedding dimension
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), # half size
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # half size
        # _make_res_block produce a ResBlock
        self.layer0 = self._make_res_block(Resblock, 64, Resblocknumlist[0], stride=1)
        self.layer1 = self._make_res_block(Resblock, 128, Resblocknumlist[1], stride=2) # half size
        self.layer2 = self._make_res_block(Resblock, 256, Resblocknumlist[2], stride=2) # half size
        self.layer3 = self._make_res_block(Resblock, 512, Resblocknumlist[3], stride=2) # half size
        self.avgpool = nn.AvgPool2d(7, stride=1) # output tensor : (batch , 512 , 1 ,1 )
        self.fc = nn.Linear(512, num_classes)

    def _make_res_block(self, Resblock, planes, Resblocknum, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            ) # In general, the number of channels * 2, the size becomes 1/2
        layers = []
        layers.append(Resblock(self.inplanes, planes, stride, downsample)) #
        self.inplanes = planes
        for i in range(1, Resblocknum):
            layers.append(Resblock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('----------------start--------------')
        # input size : ( 8 , 3 , 224 ,224 ) , assume batch is 8
        x = self.conv1(x)
        # print(x.shape) : torch.Size([8, 64, 112, 112])
        x = self.maxpool(x)
        # print(x.shape) : torch.Size([8, 64, 56, 56])
        x = self.layer0(x)
        # print(x.shape) : torch.Size([8, 64, 56, 56])
        x = self.layer1(x)
        # print(x.shape) : torch.Size([8, 128, 28, 28])
        x = self.layer2(x)
        # print(x.shape) : torch.Size([8, 256, 14, 14])
        x = self.layer3(x)
        # print(x.shape) : torch.Size([8, 512, 7, 7])
        x = self.avgpool(x)
        # print(x.shape) : torch.Size([8, 512, 1, 1])
        x = x.view(x.size(0), -1)
        # print(x.shape) : torch.Size([8, 512])
        x = self.fc(x)
        # print(x.shape) : torch.Size([8, 10]) , 10 for class num

        return x


num_classes = 10
num_epochs = 20
learning_rate = 0.01

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

# Train the model
total_step = len(train_loader)
epoch_loss = 0
step = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        step += 1
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        epoch_loss += loss
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        # gc.collect()
        # 清除内存，尽量避免主动调用gc.collect()，除非当你new出一个大对象，使用完毕后希望立刻回收，释放内存

        print('Epoch [{}/{}] - Step {} - Loss: {:.4f}'
              .format(epoch + 1, num_epochs, step, loss.item()))

    epoch_loss /= total_step
    print('Epoch [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, epoch_loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         del images, labels, outputs
#
#     print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
#
#
#


