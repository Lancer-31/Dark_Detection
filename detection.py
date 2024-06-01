#!/usr/bin/env python3
import os

dirpath1 = '/home1/tjh/tjh/Dark_detection/Dark2/train/3'
targetfile = '/home1/tjh/tjh/Dark_detection/Dark2/train/3.txt'
def isimage(fn):
    return os.path.splitext(fn)[-1] in (
        '.jpg', '.JPG', '.png', '.PNG')
def main():
    imagelist1 = []
    # for r1, ds1, fs1 in os.walk(dirpath1):
    for fn1 in os.listdir(dirpath1):
        # for fn1 in fs1:
        # if not isimage(fn1):
        #     continue
        # fname1 = os.path.join(dirpath1, fn1) + ' ' + '0'
        # s1 = list(fn1)
        # if s1[-3] == 'p':
        #     continue
        # s1[-1] = 'g'
        # s1[-2] = 'n'
        # s1[-3] = 'p'
        # fn2 = ''.join(s1)
        s2 = list(dirpath1)
        s3 = s2[-7:]
        fn3 = ''.join(s3)
        # fname1 = os.path.join(r1, fn1) + ' ' + '0' + ' '+ os.path.join(r1, fn2)
        fname1 = os.path.join(fn3)+'/'+fn1+ ' '+ '3'

        print(fname1)
        imagelist1.append(fname1)
    # imagelist2 = []
    # for r2, ds2, fs2 in os.walk(dirpath2):
    #     for fn2 in fs2:
    #         if not isimage(fn2):
    #             continue
    #         fname2 = os.path.join(r2, fn2)
    #         print(fname2)
    #         imagelist2.append(fname2)
    # imagelist = []
    # for i in range(len(imagelist1)):
    #     imagelist[i] = imagelist1[i] + ' ' + imagelist2[i]

    if not imagelist1:
        print('image not found')
        return
    target = os.path.join(targetfile)
    with open(target, 'w') as f:
        f.write('\n'.join(imagelist1))
    print('the path of images have been wirte to',
          target)


if __name__ == '__main__':
    main()