# -^- coding: utf8 -^- 

ENVIRONMENT = "TOY"
# ENVIRONMENT = "BLUE"
# ENVIRONMENT = "VREP"

RESUME = 22300        #使用哪个模型
TESTBEHAVE = True   #是否是测试状态，而不是带有exploration的学习状态

#关于输入输出状态 (如果想改变env需要手动更改main里面的import)
OBSERVETOPO = True  #是否观测地形信息
REFRESHTOPO = False  #是否每一轮都更新一次地图的观察，不开启这个不能在vrep运行时拖动障碍物
FUTHERTOPO = True   #是否加入更多的observation，覆盖面积是原terrain观察的四倍
SETTEDTOPO = False  #是否每个轮更新一个随机地图

#地图设置
# MAP = "fence"     #fence 有长条形的障碍物
MAP = None
SETMAP = False       #是否把地图设置成目标地图，用于blue

#计算过程有关选项
NOISE = False       #是否加高斯噪声
CLIP = False         #是否设定一个迈步长度的最大值，用于blue
LARGEMODEL = True   #使用大模型


# 输出选项
COMMENT="BLUE"      #用于作为log文件夹内tlogger输出的名字
FILEOUT = False     #把所有的print输出重定向放在文件里面
FILEOUT = FILEOUT and ENVIRONMENT=="TOY"
LOGGING = False     #记录tensorboard以及我的textlogger
DISPLAY_OBS= False    #是否在env层面把俯视图通过matplotlib画出来


# toyrep检测合格姿势选项
STRICT_BALANCE = True   #重心在三个落脚点的某一个区域
TIPS_order = False      #每个脚在自己严格的60度范围内
TIPS_distance = True    #每两个脚的夹角不能太小
SAFE_ANGLE = False
FOOTRANGE = None #[0.2,2.25] # 为了方便计算，存个平方数吧 #[0.5,1.5]
DISPLAY = False         #toyrep是否可视化（3D的哦）



#Reward Options:
POSITIVEREWARD = False #这样取一个e，可以让agent尽量学会存活 想法：curriculum学习是不是应该出了常更换任务之外还要常更换critic
RWD_PAIN = True         #腿过长penalty
RWD_DANEROUS = False    #离障碍太近 penalty #已删除
RWD_BALANCE = True
RWD_TORQUE = True
RWD_PASS_WALL =  True

#Reward factors:
RWDFAC_PAIN = 0.1
RWDFAC_DANEROUS = 1
RWDFAC_BALANCE = 1
RWDFAC_TORQUE = 1
RWDFAC_PASS_WALL = 1

# check
BLUEROBOT = ENVIRONMENT=="BLUE" 
import platform
FCNTL = platform.system()=="Linux"
assert(not (FUTHERTOPO and not OBSERVETOPO))
assert(not (SETMAP and SETTEDTOPO ))