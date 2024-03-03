import csv, torch, os
import numpy as np
from torch import nn

def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return f1

def stats_report(mylist):
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'
    return output

def load_csv(csvf):
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename,  row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float64)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['data']
        return {
            'data': torch.from_numpy(text),
            'label': sample['label']
        }


class CustomDataset():
    def __init__(self, root_dir, indice_dir, mode, size, subject_id=None, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = load_csv(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            if subject_id is not None and k.startswith(subject_id):
                self.names_list.extend([f"{k} {filename}" for filename in v])
            elif subject_id is None:
                self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + ' does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'data': IEGM_seg, 'label': label}

        if self.transform:
            sample = self.transform(sample)
            # sample = self.transform(sample)

        return sample



def pytorch2onnx(net_path, net_name, size):
    net = torch.load(net_path, map_location=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, size, 1)

    optName = str(net_name)+'.onnx'
    torch.onnx.export(net, dummy_input, optName, verbose=True)
    
def save_model_and_stats(model, stats, model_name = "net", stats_save_name = "stats"):
    train_loss, train_acc = stats[0], stats[1]
    test_loss, test_acc = stats[2], stats[3]
    
    torch.save(model, f'./saved_models/{model_name}.pkl')
    torch.save(model.state_dict(), f'./saved_models/{model_name}_state_dict.pkl')
    
    save_dir = f"./saved_models/{stats_save_name}.txt"
    file = open(save_dir, 'w')
    file.write("Train_loss\n" + str(train_loss) + "\n\n")
    file.write("Train_acc\n" + str(train_acc) + "\n\n")
    file.write("Test_loss\n" + str(test_loss) + "\n\n")
    file.write("Test_acc\n" + str(test_acc) + "\n\n")
    
class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std  
    
    def __call__(self, x):
        p = torch.rand((x.shape[0],1,1,1),device=x.device)
        res = x + torch.randn(x.shape) * self.std + self.mean
        return torch.where(p >= 0.3, x, res)

class FlipSeg:
    def __call__(self, x:torch.Tensor):
        d = torch.rand((x.shape[0],1,1,1),device=x.device)
        return torch.where(d >= 0.5, x, x.flip(2))
