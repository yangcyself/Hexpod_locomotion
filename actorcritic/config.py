# -^- coding: utf8 -^- 

# ENVIRONMENT = "TOY"
ENVIRONMENT = "BLUE"
# ENVIRONMENT = "VREP"

COMMENT="BLUE"
FILEOUT = False
FILEOUT = FILEOUT and ENVIRONMENT=="TOY"

REFRESHTOPO = True  #是否每一轮都更新一次地图的观察，不开启这个不能在vrep运行时拖动障碍物
# FUTHERTOPO = False
FUTHERTOPO = True   #是否加入更多的observation，覆盖面积是原terrain观察的四倍
# SETTEDTOPO = True
SETTEDTOPO = False  #是否每个轮更新一个随机地图
DISPLAY_OBS=True    #是否在env层面把俯视图通过matplotlib画出来
# MAP = "fence"     #fence 有长条形的障碍物
MAP = None
NOISE = False       #是否加高斯噪声
SETMAP = True       #是否把地图设置成目标地图，用于blue
CLIP = True         #是否设定一个迈步长度的最大值，用于blue
LARGEMODEL = True   #使用大模型



#Configues for topomain
RESUME = 22300      #使用哪个模型
# RESUME = 3400
# RESUME = "hopior/22000"
LOGGING = False     #记录tensorboard以及我的textlogger
OBSERVETOPO = True  #是否观测地形信息
TESTBEHAVE = True   #是否是测试状态，而不是带有exploration的学习状态


#Configurations in toyrep
STRICT_BALANCE = True   #重心在三个落脚点的某一个区域
TIPS_order = False      #每个脚在自己严格的60度范围内
TIPS_distance = True    #每两个脚的夹角不能太小
DISPLAY = False         #toyrep是否可视化（3D的哦）

BLUEROBOT = ENVIRONMENT=="BLUE" 
import platform
FCNTL = platform.system()=="Linux"


#Reward Options:
RWD_PAIN = True         #腿过长penalty
RWD_DANEROUS = False    #离障碍太近 penalty


#Reward factors:
RWDFAC_PAIN = 0.01
RWDFAC_DANEROUS = 1



assert((SETMAP and not SETTEDTOPO )or (not SETMAP and SETTEDTOPO))