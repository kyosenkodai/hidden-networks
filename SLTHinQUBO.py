# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

torch.backends.cudnn.enabled = False

args = None

class GetSubnet(autograd.Function):# autograd.Functionを継承して、GetSubnetクラス定義 スコアに基づいてサブネットを取得するためのカスタム関数
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort() #scoreをフラット化してソート 
        # .sort()は ソートした値, 元のインデックスを返すがインデックスのみ使用 (3,1,2) -> (1,2,3), (1,2,0) となる 
        j = int((1 - k) * scores.numel()) #socres.numel()はテンソルの要素数を返すため1-k%のインデックスを計算

        # flat_out and out access the same memory.　#シャローコピー
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out
        #返り値は、スコアの上位k%に対応する要素が1、それ以外が0のマスクテンソル score形状

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass. マスクは勾配に影響を与えない
        return g, None
    


class SupermaskConv(nn.Conv2d): # conv2dを継承して、SupermaskConvクラス定義 スコアに基づいてサブネットを取得するためのカスタム畳み込み層
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size())) #スコア用に重みと同じ形のテンソルを作成し学習パラメータとして登録　*TODO
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5)) #スコアの初期化をHeの一様分布で行う　(平均0, 標準偏差sqrt(2/fan_in))の一様分布　*TODO のためちゃんと理解したほうがよさそう

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu") # 重みの初期化 ほんとに大事

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False #重み更新処理の無効化 これしないと変更されるため実装時注意しよう

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity) #sparsityはtop k%のk  *TODO  絶対値いるか？スコア更新時にきにかけておく必要ありそう
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        # xは畳み込み層出力
        return x

class SupermaskLinear(nn.Linear): # nn.Linearを継承して、SupermaskLinearクラス定義 スコアに基づいてサブネットを取得するためのカスタム全結合層
    #畳み込み層と同じようにスコアを使って重みをマスクする
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
        return x
   
class GetSubnetSecondOrder(autograd.Function):# autograd.Functionを継承して、GetSubnetクラス定義 スコアに基づいてサブネットを取得するためのカスタム関数
    @staticmethod
    def forward(ctx, scores, k, hidden_layer):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()

        _, idx = scores.flatten().sort() #scoreをフラット化してソート 
        # .sort()は ソートした値, 元のインデックスを返すがインデックスのみ使用 (3,1,2) -> (1,2,3), (1,2,0) となる 
        j = int((1 - k) * scores.numel()) #socres.numel()はテンソルの要素数を返すため1-k%のインデックスを計算

        # flat_out and out access the same memory.　#シャローコピー
        flat_out = out.flatten()
        flat_out.reshape(-1, hidden_layer)
        for i in range(hidden_layer):
            if i == 0:
                flat_out[idx[:j]] = 0
                flat_out[idx[j:]] = 1
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out
        #返り値は、スコアの上位k%に対応する要素が1、それ以外が0のマスクテンソル score形状

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass. マスクは勾配に影響を与えない
        return g, None
 

class SecondOrderSupermaskLinear(nn.Linear): # nn.Linearを継承して、SupermaskLinearクラス定義 スコアに基づいてサブネットを取得するためのカスタム全結合層
    #畳み込み層と同じようにスコアを使って重みをマスクする
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.solo_scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.interaction_scores = nn.Parameter(torch.Tensor(self.weight.size())) # 初期値0なのでこのまま
        nn.init.kaiming_uniform_(self.solo_scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnetSecondOrder.apply(self.solo_scores.abs(), args.sparsity, args.hidden_layer)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)

# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d): #いったんここは使わないが、非アフィン正規化を使用するためのクラス? よくわからん　バッチノーマライゼーションしません
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # SupermaskConv: in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        # SupermaskLinear: in_features, out_features, bias
        self.conv1 = SupermaskConv(3, 32, 3, 1, bias=False) 
        self.conv2 = SupermaskConv(32, 64, 3, 1, bias=False)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc0 = SupermaskLinear(64 * 14 * 14, args.linear_size, bias=False)  #変更箇所 入力と出力が一致する隠れ層を何段か追加
 
        self.fcList = nn.ModuleList()  # nn.ModuleListを使って可変長の層を保持するリストを作成
        for layer in range(args.hidden_layer):
            self.fcList.append(SecondOrderSupermaskLinear(args.linear_size, args.linear_size, bias=False))  #それぞれが層
        self.fcLast = SupermaskLinear(args.linear_size, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.dropout2(x)
        for fc in self.fcList:
            x = fc(x)
            x = F.relu(x)
            x = self.dropout2(x)
        x = self.fcLast(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 QUBO')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    parser.add_argument("--linear-size", type=int, default=128, metavar="N",
                        help="linear size (default: 128)")
    parser.add_argument('--hidden-layer', type=int, default=4, metavar='N',
                        help='number of hidden layers (default: 4)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(args.data, 'cifar10'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(args.data, 'cifar10'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(args).to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    criterion = nn.CrossEntropyLoss().to(device)   # 損失関数の設定、今回はクロスエントロピー誤差を使用
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, criterion, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn_qubo.pt")


if __name__ == '__main__':
    main()
