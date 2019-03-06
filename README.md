# 可爱小蓝机器人的控制代码

## 组成

### 目录结构：
1. **根目录下面的文件**：主要负责和底层机器人或者仿真环境的接口
   1. blueGait
   2. bluerep 两个blue文件开头代表和小蓝机器人交互的步态和接口
   3. SlamListener 在使用小蓝机器人时，收听slam信息
   4. powerGait 和vrep或者toyrep交互的接口
   5. toyrep 为了简化仿真模拟的vrep
   6. vrep  vrep环境的接口
   7. UDPerperiment 没用
   8. log   没用
   9. vec_rot 是slamlistener生成的临时文件，没用
   10. remoteApi和vrepConst， vrep提供的接口，不用管
2.  **tcd** : 田畅达的qlearning_easier 以及姚青山的kinetic
3.  **scene**：所有的 vrep地图文件
    1.  _3: 没有反解的模型
    2.  _4: 有反解的模型，没有障碍物，只有目标
    3.  _4_with_barrier 散乱的障碍物，需要在训练过程中重新刷新位置
    4.  _4_test 
    5.  _4_setted: 两套人工设计的地图
4.  **lowrep**: 针对LLC的文件
    1.  FootControl 没用了
    2.  lowGait 使用LLC的步态，和powerGait姿势一样，只是需要调用和参考LLC，并且针对ttt3
    3.  LLController 一个由神经网络构成的反解器
5.  **actorcritic** HLC， 因为使用DDPG方法训练由此得名
    1.  logs\ 用于存放tensorboard 的log，以及训练时统计的各种原因的penalty数据
    2.  Models\ 存放模型
    3.  AcceptionCheck：一个脚本用于比较不同模型的好坏,打开vrep,执行这个脚本,脚本就可以加载不同模型,然后跑一遍,记录分数到一个pkl文件里
    4.  config： 所有的控制量，通过这个文件控制加载模型，使用哪个环境，以及一些reward添加项等等等等
    5.  logger： tensorboard的相关封装以及我手写的文字记录,我实现的tlogger统计着爆炸的原因,各种扣分原因和比例
    6.  oldenv： 使用最开始的reward 方法的 env，计算reward，提供observation ，调用下面的gait，被main调用,但是使用的action是机器人脚的位置加机器人身体变化位置
    7.  toyenv： 使用deeploco论文里面reward的env，因为发现不如oldenv好用，所以功能不如oldenv全
    8.  finalenv: 使用最开始的reward 方法的 env，计算reward，提供observation, action是默认机器人身体水平的.
    9.  toymain：没有地图观测的main函数
    10. topomain：有地图观测的main函数(更完善)
    11. train.py, utils.py, buffer.py, model.py, largermodel.py: 和DDPG训练有关的代码，Largermodel的模型更大
6.  **tf_DDPG** tensorflow 实现的DDPG代码
    1.  pendulum: 原始的pendulum
    2.  pendulummain: 用来执行DDPG训练的main函数 


### 调用关系
1. **main调用env**:上层传递的动作为腿的相对身体的目标xyz位置。 env提供reset， step 等功能，在这里计算observation和 reward
2. **env调用Gait** ：Gait执行上层命令，分解把腿移过去的动作，同时包含所有仿真环境中的物体句柄的定义
3. **Gait调用rep** Gait控制模拟或者真实的环境。


## 运行

### 运行使用Pytorch实现的DDPG训练
cd 到actorcritic 文件夹

用python 运行`topomain.py`

### 运行使用Pytorch实现的TRPO训练
cd 到actorcritic 文件夹

用python 运行`trpomain.py`

### 运行使用Tensorfl实现的DDPG训练
cd tf_DDPG 文件夹

用python 运行`pendulummain.py`

### 参数配置
见config.py
由于大部分开发都是基于topomain的, 所以有的config对于其他的main无效

## p.s.
经过测试,两个基于pytorch的main代码用来训练一个倒立摆都比较困难,收敛速度远不如tf.所以我们最后使用的tf方法