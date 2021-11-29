from torch.utils.data.dataloader import DataLoader
from dataset import DataSet
from network import NetA, SubCNNA, NetA4
from torch import optim
from torch import nn, save, load, Tensor
from collections import Counter
from util import calculate_mcc
import time


def train(epoch, batch_size=128, lr=0.00001):
    model.cuda()
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    print("=========Epoch-{}==========[lr:{}]".format(epoch, lr))
    dataset = DataSet("train_data_a.json", True)
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    tloss = 0
    avg_loss = 0
    start_time = time.time()
    for i, (left, right, label, angle) in enumerate(train_data):
        optimizer.zero_grad()
        left = left.cuda()
        right = right.cuda()
        label = label.cuda()
        angle = angle.cuda()
        preds = model(left, right, angle)
        loss = lossfn(preds, label)
        loss.backward()
        optimizer.step()
        tloss += loss
        avg_loss = round(float(tloss / i), 3)
        curtime = time.time()
        eta = (curtime - start_time) / (i + 1) * (len(train_data) - i)
        print('\r', "{}/{} current loss {}, avg loss {}, ETA {}s."
              .format(i * batch_size, len(dataset), loss, avg_loss, round(eta, 2)), end="")
    print("\nEpoch {} has finish. Epoch loss is {}".format(epoch, avg_loss))


def testCHK(**kwargs):
    chkname = kwargs.get("chkname")
    epoch = kwargs.get("epoch")
    model.eval()
    model.cuda()
    if chkname is not None:
        fchk = open(chkname, 'rb')
        model.load_state_dict(load(fchk))
        print("=========Test-{}==========".format(chkname))
    else:
        print("=========Test-{}==========".format(epoch))
    test_data = DataLoader(DataSet("test_data_a.json"), batch_size=128, shuffle=False)
    res = []
    corr = []
    for i, (left, right, label, angle) in enumerate(test_data):
        left = left.cuda()
        right = right.cuda()
        label = label.cuda()
        angle = angle.cuda()
        pred: Tensor = model(left, right, angle)
        res.extend(pred.tolist())
        corr.extend(label.tolist())
    print(res[2])
    res = list(map(softmax_convert, res))
    corr = list(map(softmax_convert, corr))
    tp, tn, fp, fn = 0, 0, 0, 0
    for x in range(len(res)):
        if res[x] == corr[x]:
            if corr[x] == [0, 1]:
                tp += 1
            if corr[x] == [1, 0]:
                tn += 1
        elif res[x] != corr[x]:
            if corr[x] == [0, 1]:
                fn += 1
            if corr[x] == [1, 0]:
                fp += 1
    print("labels: " + str(corr[:10]))
    print("result: {}".format(res[:10]))
    print(Counter(map(tuple, corr)))
    print(Counter(map(tuple, res)))
    print("tp:{},tn:{},fp:{},fn:{}".format(tp, tn, fp, fn))
    print("Accuracy:{}/{}".format(tp + tn, tp + tn + fn + fp))
    print("Precision:{}/{}".format(tp, tp + fp))
    print("Recall:{}/{}".format(tp, tp + fn))
    print("MCC:{}".format(calculate_mcc(tp, fp, fn, tn)))


def softmax_convert(tar):
    if tar[0] == max(tar):
        return [1, 0]
    else:
        return [0, 1]


def sigmod_convent(tar):
    if tar[0] >= 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    model = NetA(SubCNNA)
    model.cuda()
    init_lr = 0.0001
    # for e in range(1, 82):
    #     if e > 1 and (e - 1) % 20 == 0:
    #         init_lr *= 0.5
    #     train(e, lr=init_lr, batch_size=256)
    #     save(model.state_dict(), open("{}a_epcoh-checkpoint.chk".format(e), 'wb'))
    #     testCHK(epoch=e)
    testCHK(chkname="62a_epcoh-checkpoint.chk")
