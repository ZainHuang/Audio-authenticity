
from glob2 import glob
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from models import *
import argparse
import joblib
from sklearn.model_selection import StratifiedKFold
from feature import *

parser = argparse.ArgumentParser(description = 'resnet34 base on pytorch Training')
parser.add_argument('--ACTION' , default = 'train' , help = '输入data用于生成train和test数据；输入train用于训练模型；输入test用于预测结果')

args = parser.parse_known_args()[0]
action = args.ACTION


#+++++++++++++++++++++++++++++++++++++++++++++++++++

model_to_pred = ['001']

#+++++++++++++++++++++++++++++++++++++++++++++++++++

feature_path = '../user_data/feature' # 特征存放文件夹
model_path = '../user_data/model' #模型存放文件夹


val_path = "../user_data/val_acc.txt"
bastacc_path = "../user_data/best_acc.txt"

df = pd.read_table('../data/train/train.txt', sep=" ")
Speaker_ID_dict = {}
k = 0
for i in df.Speaker_ID.unique():#[df.Is_Faked==0]
    Speaker_ID_dict[i] = k
    k += 1
pred_to_label = dict(zip(Speaker_ID_dict.values(), Speaker_ID_dict.keys()))


BATCH_SIZE = 50
seed = 2020
np.random.seed(seed)
torch.manual_seed(seed)#为CPU设置随机种子
torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic =True


class SentenceDataSet(Dataset):
    def __init__(self, sent, sent_label,transform):
        self.sent = sent
        self.sent_label = sent_label
        self.transform = transform
    def __getitem__(self, item):
        if self.transform:
            self.sent = self.transform(self.sent)
        return self.sent[item], self.sent_label[item]

    def __len__(self):
        return len(self.sent)



def collate_fn2(batch_data):
    sent_seq = [joblib.load(xi[0]) for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    label = torch.tensor(label)
    return sent_seq,label




def train_model(model, xdf, epoch):
    model.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    k = 0
    for i in xdf:
        k += 1
        star_time = time.time()

        img, label = torch.Tensor(i[0]).to('cuda', non_blocking=True), i[1].to('cuda', non_blocking=True)

        img = img.reshape(img.size(0), 1, img.size(1), img.size(2))

        data, labels_a, labels_b, lam = mixup_data(img, label, 0.1)

        oof = model(data)

        loss = mixup_criterion(loss_function, oof, torch.tensor(labels_a, dtype=torch.long).cuda(),
                               torch.tensor(labels_b, dtype=torch.long).cuda(), lam)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        sum_loss += loss.item()
        _, predicted = torch.max(oof.data, 1)  # [0]

        total += len(label)
        correct += (predicted == label).sum()

        print('Trainding model at EPOCH:%d | Loss: %.03f | Acc: %.3f%% | total:%d | correct:%d'
              % (epoch + 1, float(sum_loss) / k, 100 * correct / total, total, correct))
    return model, sum_loss, correct, total


def val_model(model, xval, epoch):
    with torch.no_grad():
        model.eval()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        sum_label = 0.0
        # best_epoch = 0.0
        k = 0

        for i in xval:
            star_time = time.time()

            k += 1
            img, label = torch.Tensor(i[0]).cuda(), i[1].cuda()
            img = img.reshape(img.size(0), 1, img.size(1), img.size(2))

            oof = model(img)

            _, predicted = torch.max(oof.data, 1)
            total += len(label)
            loss = loss_function(oof, label.long())
            sum_loss += loss.item()
            correct += (predicted == label).sum()
            print(
                'vaild score EPOCH:%s LOSS:%03f | Acc:%.3f%% | total:%d | correct:%d | spend_time:%d'
                % (epoch + 1, float(sum_loss) / k, 100.0 * correct / total, total, correct,
                   (time.time() - star_time)))
    return model, sum_loss, correct, total


def pred_test(model, ml, model_path, n, xtest):
    print(ml)
    p = np.zeros((n, 1000))
    model_path
    for ii in ml:
        print("加载模型权重")
        path = model_path + '/model_%s.pth' % ii
        print(path)
        model.load_state_dict(torch.load(path));
        model.eval();
        k = 0

        with torch.no_grad():
            for i in tqdm(xtest):
                k += 1
                img, label = torch.Tensor(i[0]).cuda(), i[1].cuda()
                img = img.reshape(img.size(0), 1, img.size(1), img.size(2))
                pred = model(img)

                if k == 1:
                    x = pred
                else:
                    x = torch.cat([x, pred])

        x = x.cpu().detach().numpy()
        p += x / len(ml)
    return p


if __name__ == '__main__':
    if action == 'feature':
        pool_feature('train')
        pool_feature('test')

    elif action == 'train':
        # 模型训练======================================================================================

        train = np.array([i for i in glob(feature_path + '/train/*')])
        y = np.array([i.split('_')[2] for i in train])
        y = pd.DataFrame({'Audio_Name': y}).merge(df, how='left')

        label_Speaker_ID = y['Speaker_ID'].map(Speaker_ID_dict)
        y = label_Speaker_ID
        xall = SentenceDataSet(train, [i for i in y], None)
        xall = DataLoader(xall, batch_size=BATCH_SIZE, num_workers=6, shuffle=True, collate_fn=collate_fn2)

        print(pd.Series(y).unique(), len(pd.Series(y).unique()))

        best_acc = -1
        EPOCH = 20
        with open(val_path, "w") as f:
            with open(bastacc_path, "w") as f3:
                kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
                kfold = kfolder.split(train, y)
                for ki, (tn_index, val_index) in enumerate(kfold):
                    print('fold_%d' % ki)
                    if ki > 0:
                        pass
                    else:

                        model = se_resnet34().cuda()
                        model.conv1 = nn.Conv2d(1, 64, kernel_size=(10, 5), stride=1, padding=(0, 0), bias=False).cuda()
                        optimizer = Ranger(model.parameters())
                        loss_function = ArcLoss(1000).cuda()
                        best_acc = 0

                        for epoch in range(EPOCH):
                            epoch = epoch + ki * 100
                            model, sum_loss, correct, total = train_model(model, xall, epoch)

                            print('Saving model......')
                            torch.save(model.state_dict(), model_path + '/model_%03d.pth' % (epoch + 1))
                            localtime = time.asctime(time.localtime(time.time()))

                            f.write(
                                'vaild score EPOCH:%s LOSS:%.3f | Acc:%.3f%% | total:%d | correct:%d | time:%s'
                                % (epoch + 1, float(sum_loss) / k, 100.0 * correct / total, total, correct,
                                   localtime))
                            f.write("\n")
                            f.flush()

                            if 100 * correct / total > best_acc:
                                best_acc = 100 * correct / total
                                best_epoch = epoch + 1

                            print(" EPOCH: %s  | best_epoch : %03d | best_acc ：%.3f%% " % (
                            epoch + 1, best_epoch, best_acc))
                            if epoch - best_epoch >= 15:

                                print('Saving model......')
                                torch.save(model.state_dict(), model_path + '/model_%03d.pth' % (epoch + 1))
                                localtime = time.asctime(time.localtime(time.time()))

                                f3.write(
                                    "best_epoch : %03d | best_acc ：%.3f%%  | time:%s"
                                    % (best_epoch, best_acc, localtime))
                                f3.write("\n")
                                f3.flush()
                                break

                            if correct == total:

                                print('Saving model......')
                                torch.save(model.state_dict(), model_path + '/model_%03d.pth' % (epoch + 1))
                                localtime = time.asctime(time.localtime(time.time()))

                                f3.write(
                                    "best_epoch : %03d | best_acc ：%.3f%%  | time:%s"
                                    % (best_epoch, best_acc, localtime))
                                f3.write("\n")
                                f3.flush()
                                break

    else:
    # 预测===================================================================================

        model = se_resnet34().cuda()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(10, 5), stride=1, padding=(0, 0), bias=False).cuda()

        test = glob(feature_path + '/test/*')  # [:1000]
        xtest = SentenceDataSet(np.array(test), [0 for i in test], None)
        xtest = DataLoader(xtest, batch_size=50, shuffle=False, num_workers=4,
                           collate_fn=collate_fn2, pin_memory=True)
        AudioID = pd.DataFrame({'AudioID':[i.split('_')[2] for i in test]})


        p = pred_test(model,model_to_pred,model_path,len(test),xtest)
        sub = pd.concat([AudioID,pd.DataFrame(p)],axis=1).groupby(['AudioID']).sum()
        sub = pd.DataFrame({'AudioID':sub.index , 'PersonID' : np.argmax(np.array(sub),axis=1)})
        sub['PersonID'] = sub['PersonID'].map(pred_to_label)
        sub['Is_Faked'] = sub['PersonID']
        sub['Is_Faked'][sub['PersonID']=='0']=1
        sub['Is_Faked'][sub['PersonID']!='0']=0
        print(sub['PersonID'].value_counts())
        print(sub['Is_Faked'].value_counts())

        sub[['AudioID','Is_Faked','PersonID']].to_csv('../prediction_result/submit20.txt', sep=" ",index=None,header=None)



