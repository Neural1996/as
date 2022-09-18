import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
"""
包含几种数据预处理函数
"""


# 几种数据预处理策略
def cv_random_flip(img, label):
    # 随机左右翻转
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    # 随机裁剪
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    # 随机旋转
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    # 图像色彩增强
    bright_intensity = random.randint(5, 15) / 10.0  # 亮度
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0  # 对比度
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0  # 色彩强度
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0  # 锐化强度
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    # 随机高斯
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    # GT加入随机噪音，gt中加噪音增加泛化性？？
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


"""
训练模型一般都是先处理 数据的输入问题 和 预处理问题 。Pytorch提供了几个有用的工具：torch.utils.data.Dataset 类和 torch.utils.data.DataLoader 类。
流程是先把原始数据转变成 torch.utils.data.Dataset 类，
随后再把得到的 torch.utils.data.Dataset 类当作一个参数传递给 torch.utils.data.DataLoader 类，得到一个数据加载器，这个数据加载器每次可以返回一个 Batch 的数据供模型训练使用。
重写Dataset该类上的方法，实现多种数据读取及数据预处理方式。
"""
# dataset for training
class PolypObjDataset(data.Dataset):
    """
    整合所有图像路径成为一个列表，读取图像，进行数据增强，进行转换
    """
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize  # 训练时的图像大小
        # 获得文件名
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]  # 如果文件是jpg格式，得到图片路径下所有图片的文件路径，组成一个list
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # sorted files
        self.images = sorted(self.images)  # sorted()函数对可迭代对象进行排序，默认升序
        self.gts = sorted(self.gts)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # 图像转换
        self.img_transform = transforms.Compose([  # 是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
            transforms.Resize((self.trainsize, self.trainsize)),  # resize到网络的输入大小
            transforms.ToTensor(),  # 转换为tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 进行数据归一化处理
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # 得到数据集的大小，图片个数
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])  # 按照图像的索引（得到的是图像的路径）读取图像
        gt = self.binary_loader(self.gts[index])  # 二值化读取
        # 数据增强
        image, gt = cv_random_flip(image, gt)  # 随机翻转
        image, gt = randomCrop(image, gt)  # 随机裁剪
        image, gt = randomRotation(image, gt)  # 随机旋转

        image = colorEnhance(image)  # 色彩增强
        gt = randomPeper(gt)  # ？gt加入噪音有什么用

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:  # with...as...语句
            img = Image.open(f)
            return img.convert('RGB')  # 图像实例对象的一个方法，用以指定一种色彩模式

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):  # 这里改多线程，shuffle代表随机打乱顺序，pin_memory锁页内存，内存的Tensor转义到GPU的显存就会更快一些
    dataset = PolypObjDataset(image_root, gt_root, trainsize)
    # data.DataLoader数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
    data_loader = data.DataLoader(dataset=dataset,  # 需要是Dataset子类
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):  # 同上
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)  # unsqueeze增加一个维度

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]  # 得到name

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'  # 变成png结尾

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
