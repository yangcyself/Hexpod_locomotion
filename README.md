# 可爱小蓝机器人的控制代码

## 组成

目录结构：
1. 根目录下面的文件

主要负责和底层机器人或者仿真环境的接口
   1. blueGait
   2. 
   3. bluerep








agent的训练有几层封装：
1. 增强学习的逻辑，涉及的文件有 `buffer.py`, `model.py`, `train.py`, `utils.py`
2. 为增强学习提供环境 `toyenv.py`。提供 reset， step 等功能，在这里计算observation和 reward
3. `toyenv.py` 调用 `powerGait.py`里面的万能步态，同时`powerGait.py`中包含所有仿真环境中的物体句柄的定义
4. `powerGait.py` 调用模拟环境，可以是`toyrep.py`, 也可以是`vrep.py` 二者接口相同。注意`powerGait.py`， 并不是完全的一层封装，`toyenv.py`也会调用 `vrep`接口

## 运行

cd 到actorcritic 文件夹

用python 运行`toymain.py`

### 图形界面配置

考虑到matplotlib的图形界面容易卡死，以及在服务器上显示不方便，可以运行`train.ipynb` 的第一个块，这样可以在浏览器中有图形界面。

开和不开图形界面速度会差很多，那个捞图形也没啥好看的，所以如果debug的话可以开着看一看，如果要训练的话就关掉吧

### 一些参数配置

选择vrep 和 toyrep 在 `powerGait.py`的import部分，注释掉不想要的就可以了，为了速度，可以相应修改N

继续前面训过的模型可以修改 toymain.py 中的RESUME

开和关掉toyrep 的图形显示 设置在 `toyrep.py`中的DISPLAY