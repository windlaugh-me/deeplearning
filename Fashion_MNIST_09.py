#!/usr/bin/env python
# coding: utf-8

# # 图像分类数据集

# ## 导入包

# In[29]:


#Fashion-MNIST数据集
#为了能在jupyter中直接显示图
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torchvision #pytorch对于计算机视觉方面的一个库
from torch.utils import data
from torchvision import transforms #对数据操作
from d2l import torch as d2l
#import log_progress

d2l.use_svg_display() #用svg显示图片，清晰度要高


# ## Fshion-MNIST数据集下载

# In[30]:



#通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中
#通过transforms.ToTensor()将图像数据从图像类型转换为float32
trans=transforms.ToTensor()
#train=True是指训练数据，transform=trans将图片转化为tensor，download=True指在线下载
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)

len(mnist_train),len(mnist_test)


# In[31]:


#黑白图片 RGB=1，第一维的0是指0example,第二维的[0]是指图片的标号
mnist_train[0][0].shape


# ## 返回Fashion-MNIST数据集的文本标签

# In[32]:


def get_fashion_mnist_labels(labels):  
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# ## 显示数据集图片

# In[33]:


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    #每个axes都是拥有自己坐标系统的绘图区域
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() #展平
     #enumerate()可遍历的数据对象组成一个索引序列，同时给出数据和数据下标
    #zip()并行遍历列表
    #ax是画板上的画布，由上下左右四个参数组成
    for i, (ax, img) in enumerate(zip(axes, imgs)): 
        #图片张量
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
            #print(ax)
        else:
            #PIL图片
            ax.imshow(img)
            #print(ax)
        ax.axes.get_xaxis().set_visible(False) #False用来隐藏坐标轴
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# ## 几个样本的图像及其相应的标签

# In[34]:


#y已经是标号了
x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
#channel就不用了，reshape成18 2行9列，get_fashion_mnist_labels(y)把标号对应的标签拿出来
show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));


# ## 读取一小批量的数据 大小为batch_size

# In[35]:


batch_size=256

#使用4个进程来读取数据
def get_dataloader_workers():
    return 0

train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

#通常模型训练很快，但是数据可能读取不过来，会看一下读取数据的时间，一般要比训练快很多
#读一次数据的时间是2.89s
timer=d2l.Timer()
for x,y in train_iter:
    continue

f'{timer.stop():.2f} sec'


# ## 下载Fashion-MNIST数据集，然后将其加载到内存中

# In[36]:


#封装一下 以便后续调用
def load_data_fashion_mnist(batch_size, resize=None):  
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for x, y in train_iter:
    print(x.shape, x.dtype, y.shape, y.dtype)
    break


# In[ ]:





# In[ ]:




