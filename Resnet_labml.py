
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

from labml  import experiment , tracker , lab ,logger , monit
from labml.configs import BaseConfigs, option, calculate, hyperparams, aggregate
from labml_helpers.device import DeviceConfigs

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

class LoaderConfigs(BaseConfigs):
    batch_size: int = 2

    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader

    data_dir : str = './data'
    Egdataset : torchvision.datasets = 'cifar10' # or 'mnist'

    if Egdataset == 'cifar10':
        print('-------------------Using CIFAR10 dataset------------------')
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                    # mean or std  can be calculated manually, but are also available online.
                )
            ]
        )
    elif Egdataset == 'mnist':
        print('-------------------Using MNIST dataset------------------')
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    else:
        print('With a custom dataset, maybe you need to specify the data transform yourself.')


def _data_loader(is_train, c: LoaderConfigs):
    if c.Egdataset == 'mnist':
        return torch.utils.data.DataLoader(
            datasets.MNIST( c.data_dir,train= is_train,download=True,transform = c.transform),
            batch_size=c.batch_size, shuffle=True )
    elif c.Egdataset == 'cifar10':
        return torch.utils.data.DataLoader(
            datasets.CIFAR10(c.data_dir, train=is_train, download=True, transform=c.transform),
            batch_size=c.batch_size, shuffle=True)
    else:
        print('Do not have any dataset !')
        raise NotImplementedError


@option([LoaderConfigs.train_loader, LoaderConfigs.valid_loader])
def data_loaders(c: LoaderConfigs):
    train_data = _data_loader(True, c)
    valid_data = _data_loader(False, c)

    return train_data, valid_data
    # automatic match LoaderConfigs.train_loader and  LoaderConfigs.valid_loader


class ResnetConfig(LoaderConfigs):
    # experiment config ,inherit from LoaderConfigs

    imgsize : int = 224
    emb_dim : int = 64
    num_classes : int
    num_epochs : int = 20
    learning_rate : float = 0.01
    layers_block : list = [3, 4, 6, 3]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
            # mean or std  can be calculated manually, but are also available online.
        )
    ])

    model : nn.Module
    criterion : nn.modules.loss = nn.CrossEntropyLoss()
    device: torch.device = DeviceConfigs()
    optimizer : torch.optim.Optimizer =  'sgd_optimizer'
    # or 'adam_optimizer' , which is come from @option() decorators defined follower
    train_log_interval = 10

    # @monit.func("Training ")  <=> with monit.section("Training"):  <=> monit.iterate("Training"): <=> monit.enum("Training")
    def train(self):
        print(f'Using {self.device} train the model !')

        self.model.train()
        with monit.section("Training "):
            tracker.set_global_step(0)  # Reset the counter

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                # **✨ Increment the global step**
                tracker.add_global_step()

                # **✨ Store stats in the tracker**
                tracker.add(
                    {'loss.train': loss}
                )
                # This stores all the loss values and writes the logs the mean on every tenth iteration.

                if tracker.get_global_step() % self.train_log_interval == 0:
                    # **✨ Save added stats**
                    # 可以理解为每10次向服务器推送一次结果
                    tracker.save()

                del data, target, output
                torch.cuda.empty_cache()

    def validate(self):
        self.model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                valid_loss += self.criterion(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        valid_loss /= len(self.valid_loader.dataset)
        valid_accuracy = 100. * correct / len(self.valid_loader.dataset)

        # **Save stats**
        tracker.save({'loss.valid': valid_loss, 'accuracy.valid': valid_accuracy})

    def run(self):
        for epoch in range(1, self.num_epochs + 1):
            self.train()
            self.validate()
            logger.log()


@option(ResnetConfig.model)
def Resnet(c : ResnetConfig):
    # If you don't specific the name of option , then it's name is the function name
    # The ResnetConfig will be pass in here , named c
    return ResNet(ResidualBlock, c.layers_block,num_classes=c.num_classes).to(c.device)


@option(ResnetConfig.optimizer)
def adam_optimizer(c: ResnetConfig):
    return torch.optim.Adam(
        c.model.parameters(), lr=c.learning_rate, momentum = 0.9
    )

@option(ResnetConfig.optimizer)
def sgd_optimizer(c: ResnetConfig):
    return torch.optim.SGD(
        c.model.parameters(), lr=c.learning_rate, weight_decay=0.001, momentum = 0.9
    )


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    conf = ResnetConfig()
    experiment.create(name='ResNet')
    experiment.configs(conf,{'optimizer': 'sgd_optimizer' ,
                        'Egdataset' : 'cifar10',
                        'num_classes' : 10 })

    seed_everything(42)
    # or   conf.set_seed.set()
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()

    # save the model
    experiment.save_checkpoint()


if __name__ == '__main__':
    main()


