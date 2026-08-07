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
        # contiguousを付けて、必ず連続テンソルにする
        out = scores.clone().contiguous()

        # 順位を取得
        _, idx = scores.flatten().sort()

        j = int((1 - k) * scores.numel())

        # outと必ずメモリを共有する1次元ビュー
        flat_out = out.view(-1)

        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out
    # @staticmethod
    # def forward(ctx, scores, k):
    #     # Get the supermask by sorting the scores and using the top k%
    #     out = scores.clone()
    #     _, idx = scores.flatten().sort() #scoreをフラット化してソート 
    #     # .sort()は ソートした値, 元のインデックスを返すがインデックスのみ使用 (3,1,2) -> (1,2,3), (1,2,0) となる 
    #     j = int((1 - k) * scores.numel()) #socres.numel()はテンソルの要素数を返すため1-k%のインデックスを計算

    #     # flat_out and out access the same memory.　#シャローコピー
    #     flat_out = out.flatten()
    #     flat_out[idx[:j]] = 0
    #     flat_out[idx[j:]] = 1

    #     return out
    #     #返り値は、スコアの上位k%に対応する要素が1、それ以外が0のマスクテンソル score形状

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
 

class SecondOrderSupermaskLinear(nn.Linear): # nn.Linearを継承して、SupermaskLinearクラス定義 スコアに基づいてサブネットを取得するためのカスタム全結合層
    #畳み込み層と同じようにスコアを使って重みをマスクする
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.solo_scores = nn.Parameter(torch.Tensor(self.weight.size())) 
        self.interaction_scores = nn.Parameter(torch.zeros(self.weight.size()[0], self.weight.size()[0], self.weight.size()[0])) #0初期化
        #interaction_scoresはx_ijk => 一層前のi からjへの辺が次のjからkへの辺に影響するかどうかを表すスコア
        nn.init.kaiming_uniform_(self.solo_scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x, pre_subnet):
        # pre_layer: s*sのマスクテンソル　x_ijは前の層のiからjへのスコア
        # masked_interaction_scores = pre_layer_mask * self.interaction_scores #(s * (s*s*s))=> s*sにする 
        # scores = self.solo_scores + masked_interaction_scores # s + (s*s) => sにする
        self.saved_input = x.detach() #順伝搬で値更新をしない 
        fixed_pre_subnet = pre_subnet.detach()
        masked_interaction_scores = torch.einsum("ji,ijk->kj", fixed_pre_subnet, self.interaction_scores) #TODO <- これでいけるらしい 後でちゃんと記法の理解
        active_count = fixed_pre_subnet.sum(dim=1).clamp_min(1.0)

        masked_interaction_scores = (
            masked_interaction_scores
            / active_count.unsqueeze(0)
        )
        scores = masked_interaction_scores*args.quad_scale +  self.solo_scores.abs()
        subnet = GetSubnet.apply(scores.detach(), args.sparsity)
        
        #以下テスト用
        with torch.no_grad():
            solo_subnet = GetSubnet.apply(
                self.solo_scores.abs(),
                args.sparsity
            )

            # 二次項を入れたことで何個のマスクが変わったか
            self.mask_diff = (
                subnet != solo_subnet
            ).sum().item()

            # 単独項と二次項の平均的な大きさ
            self.solo_mean = (
                self.solo_scores.abs().mean().item()
            )

            self.interaction_mean = (
                masked_interaction_scores.abs().mean().item()
            )
        w = self.weight * subnet
        output = F.linear(x, w, self.bias)

        # loss.backward() 後に δL/δI を取得するため
        if self.training and output.requires_grad:
            output.retain_grad()

        self.saved_output = output

        return output, subnet

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
        self.dropout2 = nn.Dropout(0.05)
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
        pre_subnet = torch.zeros_like(self.fcList[0].weight)
        for fc in self.fcList:
            x, pre_subnet = fc(x, pre_subnet=pre_subnet)  # ここではpre_layer_maskは使用しない
            x = F.relu(x)
            x = self.dropout2(x)
        x = self.fcLast(x)
        output = F.log_softmax(x, dim=1)
        return output
    
@torch.no_grad()
def set_custom_score_gradients(model):
    """
    solo:
        dL/ds_solo[k,j]
        = sum_b dL/dI[b,k] * w[k,j] * Z[b,j]

    interaction:
        dL/ds_intera[i,j,k]
        = sum_b dL/dI[b,k]
          * w_prev[j,i]
          * w_current[k,j]
          * Z_prev[b,i]
    """

    for layer_idx, current_fc in enumerate(model.fcList):
        # δL/δI_k
        # current_fcの線形出力、ReLU前の勾配
        delta_k = current_fc.saved_output.grad

        if delta_k is None:
            raise RuntimeError(
                f"fcList[{layer_idx}].saved_output.grad is None"
            )

        delta_k = delta_k.detach()          # [batch, k]

        # =================================================
        # solo_scores の勾配
        # =================================================

        z_j = current_fc.saved_input        # [batch, j]
        w_kj = current_fc.weight.detach()   # [k, j]

        solo_grad = torch.einsum(
            "bk,kj,bj->kj",
            delta_k,
            w_kj,
            z_j
        )

        solo_grad = (
            current_fc.solo_scores.detach().sign()
            * solo_grad
        )

        current_fc.solo_scores.grad = solo_grad.clone()

        # =================================================
        # interaction_scores の勾配
        # =================================================

        # fcList[0]には、その前のfcList層が存在しない
        if layer_idx == 0:
            current_fc.interaction_scores.grad = None
            continue

        previous_fc = model.fcList[layer_idx - 1]

        # 経路 i -> j -> k
        z_i = previous_fc.saved_input       # [batch, i]
        w_ji = previous_fc.weight.detach()  # [j, i]
        w_kj = current_fc.weight.detach()   # [k, j]

        # 前の層のReLU前出力
        preactivation_j = previous_fc.saved_output.detach()

        # ReLUの微分
        relu_gate_j = (
            preactivation_j > 0
        ).to(delta_k.dtype)

        interaction_grad = torch.einsum(
            "bk,bj,ji,kj,bi->ijk",
            delta_k,
            relu_gate_j,
            w_ji,
            w_kj,
            z_i
        )

        # interactionをsoloから分離する場合
        interaction_grad = (
            interaction_grad
            - interaction_grad.mean(dim=0, keepdim=True)
        )

        current_fc.interaction_scores.grad = (
            interaction_grad.clone()
        )


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        set_custom_score_gradients(model)
        # 各epochの最初のバッチだけ表示
        if batch_idx == 0:
            print(f"\n--- Epoch {epoch} second-order debug ---")

            for i, fc in enumerate(model.fcList):
                grad = fc.interaction_scores.grad

                if grad is None:
                    grad_norm = None
                else:
                    grad_norm = grad.norm().item()

                print(
                    f"fcList[{i}] "
                    f"grad_norm={grad_norm}, "
                    f"solo_mean={fc.solo_mean:.6e}, "
                    f"interaction_mean={fc.interaction_mean:.6e}, "
                    f"mask_diff={fc.mask_diff}"
                )
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
    parser.add_argument("--quad-scale",type=float,default=1.0)
    
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
