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
    3.  AcceptionCheck：一个脚本用于比较不同模型的好坏
    4.  config： 所有的控制量，通过这个文件控制加载模型，使用哪个环境，以及一些reward添加项等等等等
    5.  logger： tensorboard的相关封装以及我手写的文字记录
    6.  oldenv： 使用最开始的reward 方法的 env，计算reward，提供observation ，调用下面的gait， 被main调用
    7.  toyenv： 使用deeploco论文里面reward的env，因为发现不如oldenv好用，所以功能不如oldenv全
    8.  toymain：没有地图观测的main函数
    9.  topomain：有地图观测的main函数(更完善)
    10. train.py, utils.py, buffer.py, model.py, largermodel.py: 和DDPG训练有关的代码，Largermodel的模型更大


### 调用关系
1. **main调用env**:上层传递的动作为腿的相对身体的目标xyz位置。 env提供reset， step 等功能，在这里计算observation和 reward
2. **env调用Gait** ：Gait执行上层命令，分解把腿移过去的动作，同时包含所有仿真环境中的物体句柄的定义
3. **Gait调用rep** Gait控制模拟或者真实的环境。
4. 注意Gait并不是完全的一层封装，env也会调用vrep接口

## 运行

cd 到actorcritic 文件夹

用python 运行`topomain.py`

###参数配置
见config.py