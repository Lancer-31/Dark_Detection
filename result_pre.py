# -*-coding:utf-8-*-
import torch as t
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torchvision import transforms as T
import matplotlib.pyplot as plt
from data.data_gan import *
import option
from models.Gan_res import Newmodel
from models.Gan import PerceptualLoss, load_vgg16, define_D, GANLoss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def visualize_model(args,model):
    model.eval()

    checkpoint = torch.load("/home1/tjh/tjh/HVP/checkpoint4/ep008.pth")
    model.load_state_dict(checkpoint['state'],strict=False)

    if args.n_GPUs:
        model.cuda()
        model = nn.DataParallel(model)
    corrects_classes = 0
    y_pred = []
    y_true = []
    
    for ii, (fn, data1,data, label, input_A, input_A_gray) in enumerate(test_dataloader):
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
            
            score,fake_B,latent_realA = model(input,real_A,real_A_gray,args)
            
        preds_classes = torch.max(score, 1)[1]
        corrects_classes += torch.sum(target == preds_classes)

        plt.imshow(transforms.ToPILImage()(data1.cpu().squeeze(0)))
        m = data1.shape[3] * 0.5
        n = data1.shape[2] * 0.2
        # plt.text(m, n, 'Pred: {}  GT:{}\n'.format(pred, labels),fontdict={'size': 20, 'color': 'red'}, ha="center", va="center")
        plt.text(m, n, 'Pred: {}  GT:{}\n'.format(preds_classes.cpu().item(), label.cpu().item()), fontdict={'size': 40, 'color':  'red'}, ha="center", va="center")

        plt.axis('off')

        fn_1 = fn[0]
        img_name = fn[0].split('/')[-1].split('.')[0]
        img_name1 = fn[0].split('/')[-2].split('.')[0]
        fn_1 = fn_1.split('/')[-1].split('.')[0]
        #fn_2 = fn_1.split('/')[-2].split('.')[0]
        plt.savefig('/home1/tjh/tjh/HVP/result_pre/yejian1/'+fn_1+'.jpg')


        plt.draw()
        plt.pause(0.1)
        plt.close()

    acc_classes = corrects_classes.double() / len(test_data)
    print('Acc_classes: {:.2%} '.format(acc_classes))

if __name__ == '__main__':


    args = option.args

    model = Newmodel()
    #model = AlexNet()
    if args.n_GPUs: model.cuda()

    transform = T.Compose([
        T.Resize((384, 256)),
        #T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])


    test_data = MyDataset_pre(dir= args.dir_data, txt= args.dir_data + 'test.txt', transform=transform,
                          target_transform=transforms.ToTensor(),train=True)
    test_dataloader = DataLoader(test_data, batch_size=1,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    # for epoch in range(args.epochs):
    visualize_model(args,model)