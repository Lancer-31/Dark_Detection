import os
import torch as t
import numpy as np
from data.data_gan import *
import option
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import numpy
import torch.distributed as dist
from models.Gan_res import Newmodel
from models.Gan import PerceptualLoss, load_vgg16, define_D, GANLoss
from utils.dist_utils import dist_print
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import random
#torch.autograd.set_detect_anomaly(True)

# vis = Visdom(env = 'Net')
# vis.line([0.],[0.],win='train',opts=dict(title='train loss'))
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
    
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma = 2, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0


    
def train(args,model,criterion,optimizer,global_step,epoch,base_lr):

    model.train()
    gpu_ids = [1]
    if args.n_GPUs:
        model.cuda()
        #model = nn.DataParallel(model)
    corrects_classes = 0
    running_loss = 0.0
    loss22 = 0.0
    
    vgg_loss = PerceptualLoss(args)
    vgg_loss.cuda()
    vgg = load_vgg16("./VGGmodel", gpu_ids)
    vgg.eval()
    netD_A = define_D(args.output_nc, args.ndf,
                                args.which_model_netD,
                                args.n_layers_D, args.norm, args.no_lsgan, gpu_ids, False)
    netD_P = define_D(args.input_nc, args.ndf,
                                args.which_model_netD,
                                args.n_layers_patchD, args.norm, args.no_lsgan, gpu_ids, True)
    Tensor = torch.cuda.FloatTensor
    criterionGAN = GANLoss(use_lsgan=not args.no_lsgan, tensor=Tensor)

    # for ii, (data, label, seg_lable) in enumerate(train_dataloader):
    for ii,(data, label, input_A, input_A_gray) in enumerate(train_dataloader):
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
        
        loss = criterion(score, target)
        
        
        #w = real_A.size(3)
        #h = real_A.size(2)
        #w_offset = random.randint(0, max(0, w - args.patchSize - 1))
        #h_offset = random.randint(0, max(0, h - args.patchSize - 1))
        #fake_patch_1 = []
        #fake_patch = fake_B[:,:, h_offset:h_offset + args.patchSize,w_offset:w_offset + args.patchSize]
        #for i in range(args.patchD_3):
        #    w_offset_1 = random.randint(0, max(0, w - args.patchSize - 1))
        #    h_offset_1 = random.randint(0, max(0, h - args.patchSize - 1))
        #    fake_patch_1.append(fake_B[:,:, h_offset_1:h_offset_1 + args.patchSize, w_offset_1:w_offset_1 + args.patchSize])
        
        #pred_fake = netD_A.forward(fake_B)
        #loss_G_A = pred_fake.mean()
        
        #loss_G_A_P = 0
        #pred_fake_patch = netD_P.forward(fake_patch)
        #loss_G_A_P += criterionGAN(pred_fake_patch, True)
        #for i in range(args.patchD_3):
        #    pred_fake_patch_1 = netD_P.forward(fake_patch_1[i])
        #    loss_G_A_P += criterionGAN(pred_fake_patch_1, True)
        #loss_G_A += loss_G_A_P/float(args.patchD_3 + 1)
        
        #loss_vgg_b = vgg_loss.compute_vgg_loss(vgg, fake_B, real_A)
        #loss2 = loss_G_A + loss_vgg_b
        #loss = loss1+loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item() * input.size(0)
        #loss22 += loss2.item() * input.size(0)
        preds_classes = torch.max(score, 1)[1]
        corrects_classes += torch.sum(target == preds_classes)


    epoch_loss = running_loss / len(train_data)
    #loss22 = loss22/len(train_data)
    epoch_acc_classes = corrects_classes.double() / len(train_data)
    print('train  Loss: {:.4f}  Acc_classes: {:.2%} '.format(epoch_loss, epoch_acc_classes))
    #print('train  Loss: {:.4f} loss2:{:.4f}  Acc_classes: {:.2%} '.format(epoch_loss, loss22, epoch_acc_classes))

def save_model(net, optimizer, epoch, save_path):
    if is_main_process():
        model_state_dict = net.state_dict()
        state = {'state': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
        torch.save(state, model_path)


def test(args,model):

    model.eval()

    checkpoint = torch.load('/home1/tjh/tjh/Dark_detection/checkpoint7/ep%03d.pth' % epoch)
    model.load_state_dict(checkpoint['state'])
    # # print(checkpoint['state'])
    gpu_ids = [1]
    if args.n_GPUs:
        model.cuda()
        #model = nn.DataParallel(model)
        
    corrects_classes = 0
    y_p = np.zeros((1))
    y_t = np.zeros((1))


    for ii, (data, label, input_A, input_A_gray) in enumerate(test_dataloader):
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

        preds_classes1 = torch.max(score,1)[1]
        y_pred = preds_classes1.cpu().numpy()
        y_p = np.concatenate([y_p, y_pred], axis=0)
        y_true = target.cpu().numpy()
        y_t = np.concatenate([y_t, y_true], axis=0)
        
        corrects_classes += torch.sum(target == preds_classes1)
        

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
    if args.n_GPUs: model.cuda()

    transform = T.Compose([
        T.Resize((384, 256)),
        # T.CenterCrop((224, 224)),
        # T.Resize((320,640)),
        # T.CenterCrop((320,640)),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    criterion = t.nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': args.learning_rate}]

        optimizer = torch.optim.SGD(params,
                                    lr=args.learning_rate,momentum=args.momentum,
                                    weight_decay=args.SGD_weight_decay,nesterov=False,
                                    )
    #elif args.optimizer == 'Adam':
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': args.lr}]
    

    optimizer = torch.optim.Adam(params,
                                lr=args.lr, weight_decay=args.weight_decay,
                                amsgrad=False,)
                                  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    train_data = MyDataset_train(dir=args.dir_data, txt=args.dir_data + 'train.txt', transform=transform,
                           train=True)
    train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True)

    test_data = MyDataset_train(dir=args.dir_data, txt=args.dir_data + 'test.txt', transform=transform,
                          target_transform=transforms.ToTensor(), train=True)

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers)
    if args.resume is not None:
        dist_print('==> Resume model from ' + args.resume)
        resume_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(resume_dict['state'],strict=False)
        #if 'optimizer' in resume_dict.keys():
        #    optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(args.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    global_step = 0
    for epoch in range(resume_epoch, args.epochs):
        print('epoch {}/{}'.format(epoch, args.epochs - 1))
        train(args, model, criterion, optimizer, global_step, epoch,args.lr)
        #scheduler.step()
        save_model(model, optimizer, epoch, args.work_dir7)
        test(args, model)
