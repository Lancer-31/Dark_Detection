import torch as t
import numpy as np
import torch.nn as nn
from torch import optim
from torchvision import transforms as T
from data.data_gan import *
import option
from models.Gan_res import Newmodel
from models.Gan import PerceptualLoss, load_vgg16, define_D, GANLoss

from sklearn.metrics import confusion_matrix
def out_confusion1(cm):
    F1 = []
    Pre = []
    a = 0
    b = 0
    cm = np.array(cm)
    for i in range(4):
        Pos = 0
        Res = 0
        for j in range(4):
            pos = cm[i][j]
            res = cm[j][i]
            Pos = Pos + pos
            Res = Res + res
        tp = cm[i, i]
        c = Pos + Res
        f1 = (2 * tp / c)
        f1 = f1.astype(np.float32)
        F1.append(f1)
    F1_score = np.array(F1)
    #print(F1_score)
    w = F1_score
    return w

def test(args,model):

    model.eval()

    #checkpoint = torch.load('/home/sunyubao/hjr/xl/Deblur/HVP/checkpoint1/ep%03d.pth' % epoch)
    checkpoint = torch.load("/home1/tjh/tjh/Dark_detection/checkpoint6/ep065_87.88.pth")
    model.load_state_dict(checkpoint['state'],strict = False)

    if args.n_GPUs:
        model.cuda()

    corrects_classes = 0
    y_p = np.zeros((1))
    y_t = np.zeros((1))

    
    for ii,(data, label, input_A, input_A_gray) in enumerate(test_dataloader):
        # input = Variable(data,volatile = False)
        with torch.no_grad():
            input = Variable(data)
            target = Variable(label)
            input = input.cuda()
            target = target.cuda()
            
            input_A.resize_(input_A.size()).copy_(input_A)
            input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
            real_A = Variable(input_A)
            real_A_gray = Variable(input_A_gray)
            real_A = real_A.cuda()
            real_A_gray = real_A_gray.cuda()
            
            score = model(input,real_A,real_A_gray,args)
            #score,fake_B,latent_realA = model(input,real_A,real_A_gray,args)
            
            #loss1 = criterion(score, target)

        preds_classes = torch.max(score,1)[1]
        
        y_pred = preds_classes.cpu().numpy()
        y_p = np.concatenate([y_p, y_pred], axis=0)
        y_true = target.cpu().numpy()
        y_t = np.concatenate([y_t, y_true], axis=0)
        
        corrects_classes += torch.sum(target == preds_classes)

    acc_classes = corrects_classes.double() / len(test_data)
    print('Acc_classes: {:.2%} '.format(acc_classes))
    y_t = y_t[1:]
    y_p = y_p[1:]
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2, 3])
    w2 = out_confusion1(cm)
    print(w2)
    return acc_classes


if __name__ == '__main__':


    args = option.args

    model = Newmodel()

    transform = T.Compose([
        #T.Resize((896, 896)),
        T.Resize((384, 256)),
        #T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    criterion = t.nn.CrossEntropyLoss()
    test_data = MyDataset_train(dir=args.dir_data, txt= args.dir_data + 'test.txt', transform=transform,
                          target_transform=transforms.ToTensor(),train=True)
    test_dataloader = DataLoader(test_data, batch_size=4,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    #for epoch in range(7, args.epochs):
    test(args,model)