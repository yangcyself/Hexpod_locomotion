# -^- coding:utf-8 -^-
"""
使用方法：
import bluerep as vrep
N=5
别的不变

注意bluerep 和toyrep之间坐标系的区别：
    x，y需要交换，取相反数
    这一变换在rep与机器人的接口中实现
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from pexpect import pxssh
import math
import numpy as np
import time
# import platform
# if(platform.system()=="Linux"):
#     import fcntl
import slamListener
from actorcritic.config import *
from socket import *


WORLD_SIZE = [-5,5]

NAME = {} # name -> handle
HANDLE = {}# handle -> obj
H_count = 0

#whether make sure that the body position is near the middle of the fixed legs
STRICT_BALANCE = True 
TIPS_order = True
DISPLAY = False
# DISPLAY = True


def Handle(obj):
    global H_count
    H_count+=1
    HANDLE[H_count]=obj
    return H_count

def topology(x,y):
    # if(1<x<=1.5 and -2<y<=2):
    #     return 1
    # if(0.4<x<0.6 and 0.45<y<0.55):
    #     return 0.1
    height = 0
    for c in CLDS:
        if((x-c.loc[0])**2+(y-c.loc[1])**2<= c.size**2):
            height = max(height,c.size)
    return height

TOPO = topology


def turnVec(vec,deg):
    """
    Get the vector after turned the vec some degree
    """
    assert(vec.shape==(3,))
    return np.array([
        vec[0]*np.cos(deg) - vec[1]*np.sin(deg),
        vec[0]*np.sin(deg) + vec[1]*np.cos(deg),
        vec[2]
    ])

def distance(vec1,vec2):
    return np.sqrt(np.sum(((vec1-vec2)**2)))

def diff_ang(vec1,vec2):
    ang = math.atan2(vec1[1],vec1[0])-math.atan2(vec2[1],vec2[0])
    if(ang<-math.pi):
        ang+= math.pi
    elif (ang>math.pi):
        ang-=math.pi
    return ang

def ave_ang(angs):
    angs = np.array(angs)
    x = np.cos(angs)
    y = np.sin(angs)
    return math.atan2(np.sum(y),np.sum(x))

    



class rep_obj:
    def __init__(self,name,parent):
        if(type(name)!=list):
            name=[name]
        self.handle=[]
        for n in name:
            h = Handle(self)
            self.handle.append(h)
            NAME[n] = h
        self.parent = parent
        self.state = True

class Botnode(rep_obj):
    def __init__(self,loc,name,parent):
        self.p=np.array(loc) #the position in BCS
        assert(self.p.shape==(3,))
        super(Botnode, self).__init__(name,parent)
        # print(self.p,parent.toGlob(self.p))
        self.parent = parent
        self.loc = parent.loc + parent.toGlob(self.p) # the position in -1
        
    def reset(self,loc):
        self.p=np.array(loc)
        self.loc = self.parent.loc + self.parent.toGlob(self.p)
        self.state = True

    def fixed(self):
        return self.loc[2] <= TOPO(self.loc[0],self.loc[1])+0.005

class Hexpod(rep_obj):
    def __init__(self,reset = False):
        # self.bodyp = Botnode([0,0,0.6])
        # self.bodyo = 0 #orientation
        self.ori = 0
        height = 4.4652e-01
        self.loc = np.array([0,0,height])
        if(not reset):
            self.loc_sol = [] # the location get solving one constraints, for debugging
            self.ori_sol = []
            super(Hexpod, self).__init__("BCS",None)
            self.tips=[]
        self.state = True
        tip_start_loc = [[ 5.27530670e-01 , 3.04633737e-01 ,-height],
                            [-2.28881836e-05 , 6.09106421e-01 ,-height],
                            [-5.27527809e-01 , 3.04500699e-01 ,-height],
                            [ 5.27622223e-01 ,-3.04508090e-01 ,-height],
                            [ 1.44958496e-04 ,-6.09171569e-01 ,-height],
                            [-5.27413368e-01 ,-3.04654479e-01 ,-height]]
        for i,loc in enumerate(tip_start_loc):
            if(not reset):
                self.tips.append(Botnode(loc,['TipTarget'+str(i+1),'Tip'+str(i+1)],self))
            else:
                self.tips[i].reset(loc)

    def reset(self):
        self.__init__(reset=True)

    def toGlob(self,vec):
        return turnVec(vec,self.ori)


    def printState(self):
        print("t.loc | t.p")
        for t in self.tips:
            print(t.loc,t.p)
        print("self.loc",self.loc,"self.ori",self.ori)

    def refresh(self):
        #solve the position of body

        # refresh the height
        pass
    


class Goal(rep_obj):
    def __init__(self):
        self.loc = np.array([5,5,5])
        super(Goal, self).__init__("Goal",None)
    
    def refresh(self):
        pass
    def draw(self,ax):
        pass
    def collision_check(self):
        pass
    def reset(self):
        pass


class Cylinder(rep_obj):
    def __init__(self,size,name):
        self.loc = np.zeros(3)
        self.loc[2] = size/2
        self.size = size/2
        self.height = size
        super(Cylinder, self).__init__(name,None)

    def refresh(self):
        pass
    def draw(self,ax):
        pass
    def collision_check(self):
        pass
    def reset(self):
        pass



        

"""
Blue-rep environment
"""
hexpod = Hexpod()
goal = Goal()
OBJS = [hexpod,goal]
CLDS = []
for i in range(0, 10):
    CLDS.append(Cylinder( 0.1 , 'Barrier' + str(i)))
for i in range(0, 6):
    CLDS.append(Cylinder( 0.5 , 'Wall' + str(i)))


class robot_client:
    def __init__(self):
        HOST ='192.168.2.100'
        PORT = 5866
        self.BUFFSIZE=2048       
        ADDR = (HOST,PORT)
        self.tctimeClient = socket(AF_INET,SOCK_STREAM)
        self.tctimeClient.connect(ADDR)
        # self.command("start",[])
        # time.sleep(2)
        # self.command("en",[])
        # time.sleep(2)
        # self.command("hm",[])
        # time.sleep(5)
        # self.command("rc",[])
        # time.sleep(2)
        
    def command(self, command ,args):
        data = command+ " " + " ".join(args)
        print(data)
        # The following line is taught by 学长李逸飞
        self.tctimeClient.send((chr(len(data)+1)+chr(0)*7+chr(1)+chr(0)*31+data+chr(0)).encode())
        res = self.tctimeClient.recv(self.BUFFSIZE)[40:].decode("utf8")
        print("RES:",res)
        return res

    def __del__(self):
        # self.command("ds",[])
        # self.command("exit",[])
        self.tctimeClient.close()

class fake_robot_client:
    def command(self,command,args):
        print(command, " "," ".join(args))
        return None

"""
######################################
Set the following line when you only want to test the codes without sending commands
######################################
"""
robot  = robot_client()
# robot  = fake_robot_client()


def simxFinish(num):
    pass
def simxStart(a,b,c,d,e,f):
    return 0
def simxSynchronous(a,b):
    return 0

def simxStopSimulation(ID,opmod):
    pass
def simxStartSimulation(ID,opmod):
    for obj in OBJS:
        obj.reset() 
    return 0
    
simx_opmode_blocking = None
simx_opmode_oneshot_wait = None
simx_opmode_oneshot = None

def listTOPO():
    for b in CLDS:
        print(b.loc)

def parsePosition(res):
    """
    Parse the position in 'vec_rot.txt' and update the corresponding values
    """
    ans = np.zeros((6,2))
    lin = res[:-1].strip()
    pos = lin.split(" ")
    for i,p in enumerate(pos):
        ps = p.split(",")
        # print("p:",p)
        for j in range(2):
            # print(ps[j])
            hexpod.tips[i].p[1-j] = -float(ps[j])
            ans[i][1-j] = -float(ps[j])
    return ans


def updateRobotPosition():
    """
    vec_rot.txt is used to communicate between SLAM and bluerep
    """
    #hexpod.ori ,loc call slam
    # return
    with open("vec_rot.txt","r") as f:
        # fcntl.flock(f,fcntl.LOCK_EX)
        line = f.readline()
        nums = line.split(" ")
        while (len(nums)<6):
            line = f.readline()
            nums = line.split(" ")
        hexpod.loc[0] = -float(nums[0])
        hexpod.loc[1] = -float(nums[1])
        # print("vec_rot:",end = " ")
        # print (nums)
        hexpod.ori = float(nums[4])
        # fcntl.flock(f,fcntl.LOCK_UN)

    # for t in hexpod.tips:

    print("LOC&ORI",hexpod.loc,end = "\t")
    print(hexpod.ori)
    robot.command("gf",[])
    time.sleep(2)
    res = robot.command("gf",["-i=1"])
    # print(res)
    assert(res)

    parsePosition(res)
    
    

def display():
    """
    Display a birdview graph, use squares to represent objects.
    """
    for c in CLDS:
        x = c.loc[0]
        y = c.loc[1]
        r = c.size
        X=[x-r,x-r,x+r,x+r,x-r]
        Y=[y-r,y+r,y+r,y-r,y-r]
        ax.plot(X,Y)
    x = goal.loc[0]
    y = goal.loc[1]
    r = 0.1
    X=[x-r,x-r,x+r,x+r,x-r]
    Y=[y-r,y+r,y+r,y-r,y-r]
    ax.plot(X,Y)
    loc = hexpod.loc
    x = loc[0]
    y = loc[1]
    X=[x-r,x-r,x+r,x+r,x-r]
    Y=[y-r,y+r,y+r,y-r,y-r]
    ax.plot(X,Y)


def simxSynchronousTrigger(ID): # 每一次只在Trigger的时候更新
    pass


def simxGetObjectHandle(ID,name,opmod):
    return 1,NAME.get(name,-2)


def simxGetObjectPosition(ID, obj, cdn, opmod):
    obj = HANDLE[obj]
    if(cdn==-1):
        #CALL SLAM API
        return 1, obj.loc
    cdn = HANDLE[cdn]
    if(obj.parent==cdn):
        # CALL ROBOT API
        return 1,obj.p
    if(obj == goal):
        return 1,turnVec(obj.loc-cdn.loc,-cdn.ori)
    return 1,turnVec(obj.loc-cdn.loc,-cdn.ori)
        

def simxSetObjectPosition(ID,obj, cdn, pos ,opmod):
    obj = HANDLE[obj]
    if(not obj.state):
        return 1
    if(cdn==-1):
        obj.loc = np.array(pos)
        return 1
    cdn = HANDLE[cdn]
    if(obj.parent==cdn):
        # When to set the foot position, don't use this function
        assert (len("WRONG CALL, SET TIPS POSITION")==0)
        obj.p = np.array(pos)
        return 1
    
    
def simxGetObjectOrientation(ID,obj, cdn ,opmod):
    assert (cdn==-1)
    obj = HANDLE[obj]
    # updateRobotPosition()
    return 1,np.array([0,0,obj.ori])

def robotSetFoot(side, pee, peb):
    """
    The API of sf is:
        -i means the side
        abcdef is the x1 x2 x3 y1 y2 y3 of the foot
        ghjklm is the body's xyz and orientation 
    Because the coordinate system between bluerep layer and robot is not the same,
        here we interchange x and y and set them as their negative

    After the command is executed, update the robot configuration.
    """
    print("Begin Set Foot:", side )
    command = "sf "
    args = ["-i=%d" %side]
    for i,a in enumerate(["-d","-a","-e","-b","-f","-c"]):
        args.append(a+"="+"%.3f" % -pee[i])
    for i,a in enumerate(["-j","-g","-h","-k","-l","-m"]):
        args.append(a+"="+"%.3f" % -peb[i])
    # for i, a in enumerate(["-k","-l","-m"]):
    #     args.append(a+"="+"%.3f" % peb[i+3])
    # print(args)
    # time.sleep(5)
    robot.command(command,args)


    # update the loc of the foot and body
    hexpod.loc += turnVec(np.array(peb[0:3]),hexpod.ori)
    for t in hexpod.tips:
        t.p[0] -= peb[0]
        t.p[1] -= peb[1]

    for i in range(side,6,2):
        hexpod.tips[i].p[0] += pee[int(i/2)*2]
        hexpod.tips[i].p[1] += pee[int(i/2)*2+1]
        hexpod.tips[i].loc = hexpod.loc + hexpod.toGlob(hexpod.tips[i].p)

    for t in hexpod.tips:
        t.p = turnVec(t.p,-peb[5])

    hexpod.ori += peb[5]
    
    """
    #################
    The line before calculate the robot's configuration based on its own output
        like a openloop controler.
    To update the robot configuration from SLAM and robot APIS, use the following 
        updateRobotPosition() to cover the openloop information    
    #################
    """

    updateRobotPosition()

    if(DISPLAY_OBS and False):
        # topoobs = topoObservation()
        ax.clear()
        display()
        ax.set_xlim(-1,5)
        ax.set_ylim(-3,3)
        fig.canvas.draw()

# def drawPoint(loc):
    

if(DISPLAY_OBS and False):
    fig = plt.figure()
    ax = fig.gca()
    # obs = topoObservation()
    # ax.imshow(obs)
    display()
    # ax.autoscale([-5,5],[-5,5])
    ax.set_xlim(-1,5)
    ax.set_ylim(-3,3)
    # fig.canvas.draw()
    plt.show(block=False) 


# fig,ax = plt.subplots(1,1,projection = "3d")


if __name__ == "__main__":

    updateRobotPosition()




    
