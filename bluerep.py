# -^- coding:utf-8 -^-
"""
使用方法：
import toyrep as vrep
N=5
别的不变

注意bluerep 和toyrep之间坐标系的区别：
    x，y需要交换，取相反数
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from pexpect import pxssh
import math
import numpy as np
import time
import platform
if(platform.system()=="Linux"):
    import fcntl
import slamListener
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
        if(not reset):
            self.shape_nodes=[]
        u1_loc = [[ 1.31711960e-01,  7.59007931e-02 ,-1.78813934e-06],
                    [ 1.38759613e-04,  1.51954651e-01 ,-1.78813934e-06],
                    [-1.31512642e-01,  7.60353804e-02, -1.72853470e-06],
                    [ 1.31635189e-01, -7.60723352e-02 ,-1.84774399e-06],
                    [-1.66893005e-05, -1.51991248e-01 ,-1.90734863e-06],
                    [-1.31590843e-01, -7.59376287e-02 ,-1.78813934e-06]]
        u2_loc = [[ 0.27454329 , 0.0521549   ,0.11723149],
                    [ 0.09211826 , 0.26377678  ,0.11723149],
                    [-0.18236399 , 0.21160328  ,0.11723131],
                    [ 0.1824851  ,-0.21164024  ,0.11723143],
                    [-0.09199619 ,-0.2638135   ,0.11723149],
                    [-0.27442122 ,-0.05219185  ,0.11723137]]
        u3_loc = [[ 0.18251514 , 0.21148705  ,0.11723095],
                    [-0.0918808  , 0.26374376  ,0.11723089],
                    [-0.27433491 , 0.05223811  ,0.11723095],
                    [ 0.27445745 ,-0.05227506  ,0.11723095],
                    [ 0.09200335 ,-0.26378036  ,0.11723101],
                    [-0.18239307 ,-0.21152413  ,0.11723095]]

        for i,loc in enumerate(u1_loc):
            if(not reset):
                self.shape_nodes.append(Botnode(loc,'Hip'+str(i+1),self))
            else:
                self.shape_nodes[i].reset(loc)
            
        if(not reset):
            self.shape_nodes.append(Botnode(u2_loc[0],'J21R',self))
        else:
            self.shape_nodes[6].reset(u2_loc[0])
        for i,loc in enumerate(u2_loc[1:]):
            if(not reset):
                self.shape_nodes.append(Botnode(loc,'J21R'+str(i-1),self))
            else:
                self.shape_nodes[7+i].reset(loc)
        
        if(not reset):
            self.shape_nodes.append(Botnode(u3_loc[0],'J31R',self))
        else:
            self.shape_nodes[12].reset(u3_loc[0])
        for i,loc in enumerate(u3_loc[1:]):
            if(not reset):
                self.shape_nodes.append(Botnode(loc,'J31R'+str(i-1),self))
            else:
                self.shape_nodes[13+i].reset(loc)

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
        if(not self.state):
            return 

        self.loc_sol = []
        self.ori_sol = []
        tmin_loc = 2
        tmin_t = self.tips[0]
        for t in self.tips:
            if(t.fixed()):
                newloc = t.loc - t.p # as the height has no influence of the ang
                # if(tmin_loc>t.loc[2]):
                #     tmin_loc = t.loc[2]
                #     tmin_t = t

                self.loc_sol.append(newloc)

        #To solve the problem Error: zero-size array to reduction operation maximum which has no identity
        if(len(self.loc_sol)==0):
            print("ERROR, self.loc_sol empty \n DATAS: \n")
            self.printState()
            #set loc accoring to least leg
            self.loc_sol.append(tmin_t.loc - tmin_t.p)

        self.loc[2] = np.max(np.array(self.loc_sol),axis=0)[2]
        for t in self.tips:
            t.loc[2] = self.loc[2] + t.p[2]
        # print(self.loc[2])
        # refresh the ori
        for t in self.tips:
            if(t.fixed()):
                dif_loc = t.loc-self.loc
                # ori = math.atan2(dif_loc[1],dif_loc[0])-math.atan2(t.p[1],t.p[0])
                ori = diff_ang(dif_loc,t.p)
                self.ori_sol.append(ori)
        self.ori = ave_ang(self.ori_sol)

        # refresh the loc of all nodes
        self.loc_sol = []
        for t in self.tips:
            if(t.fixed()):
                newloc = t.loc - self.toGlob(t.p)
                self.loc_sol.append(newloc)
        self.loc = np.average(np.array(self.loc_sol),axis = 0)
        
        for t in self.tips:
            t.loc = self.loc + self.toGlob(t.p)
        for t in self.shape_nodes:
            t.loc = self.loc+ self.toGlob(t.p)
    


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
Toy-rep environment
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
        self.tctimeClient.send((chr(len(data)+1)+chr(0)*7+chr(1)+chr(0)*31+data+chr(0)).encode())
        res = self.tctimeClient.recv(self.BUFFSIZE)[40:].decode("utf8")
        print("RES:",res)
        # return "POSPOS 5e-01,3e-01 0e-01,6e-01 -5e-01,3e-01 5e-01,-3e-01 0e-01,-6e-01 -5e-01,-3e-01"
        return res
    def __del__(self):
        # self.command("ds",[])
        # self.command("exit",[])
        self.tctimeClient.close()

robot  = robot_client()


# def collision_check():
#     for obj in OBJS:
#         obj.collision_check()


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
        obj.reset() #如果有这一句,后面机器人就不会动，因为Handle在init的时候变掉了！！！！
    return 0
    # pass
simx_opmode_blocking = None
simx_opmode_oneshot_wait = None
simx_opmode_oneshot = None

def listTOPO():
    for b in CLDS:
        print(b.loc)

def parsePosition(res):
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
    #hexpod.ori ,loc call slam
    with open("vec_rot.txt","r") as f:
        # fcntl.flock(f,fcntl.LOCK_EX)
        
        line = f.readline()
        nums = line.split(" ")
        while (len(nums)<6):
            line = f.readline()
            nums = line.split(" ")
        hexpod.loc[0] = -float(nums[2])/1000
        hexpod.loc[1] = -float(nums[0])/1000
        print("vec_rot:",end = " ")
        print (nums)
        hexpod.ori = float(nums[4])
        # fcntl.flock(f,fcntl.LOCK_UN)

    # for t in hexpod.tips:

    print(hexpod.loc,end = "\t")
    print(hexpod.ori)
    robot.command("gf",[])
    time.sleep(3)
    res = robot.command("gf",["-i=1"])
    print(res)
    assert(res)

    parsePosition(res)
    
    

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
        

def simxSetObjectPosition(ID,obj, cdn, pos ,opmod):
    obj = HANDLE[obj]
    if(not obj.state):
        return 1
    if(cdn==-1):
        obj.loc = np.array(pos)
        return 1
    cdn = HANDLE[cdn]
    if(obj.parent==cdn):
        assert (len("WRONG CALL, SET TIPS POSITION")==0)
        obj.p = np.array(pos)
        return 1
    
    
def simxGetObjectOrientation(ID,obj, cdn ,opmod):
    assert (cdn==-1)
    obj = HANDLE[obj]
    # updateRobotPosition()
    return 1,np.array([0,0,obj.ori])

def robotSetFoot(side, pee, peb):
    print("Begin Set Foot:", side )
    command = "sf "
    args = ["-i=%d" %side]
    # for i,a in enumerate(["-a","-d","-b","-e","-c","-f"]):
    for i,a in enumerate(["-d","-a","-e","-b","-f","-c"]):
        # args.append(a+"="+str(pee[i]))
        args.append(a+"="+"%.3f" % -pee[i])
    for i,a in enumerate(["-j","-g","-h"]):
        # args.append(a+"="+str(peb[i]))
        args.append(a+"="+"%.3f" % -peb[i])
    for i, a in enumerate(["-k","-l","-m"]):
        args.append(a+"="+"%.3f" % peb[i+3])
    # print(args)
    # time.sleep(5)
    robot.command(command,args)

# def drawPoint(loc):
    




# fig,ax = plt.subplots(1,1,projection = "3d")

plt.show(block=False)
if __name__ == "__main__":

    updateRobotPosition()

    # n = 30

    # simxFinish(-1) # just in case, close all opened connections
    # clientID=simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to V-REP, set a very large time-out for blocking commands
    # if clientID!=-1:
    #     print ('Connected to remote API server')
    #     res = simxSynchronous(clientID, True)

    # simxStartSimulation(clientID, simx_opmode_oneshot_wait)
    # res, BCS = simxGetObjectHandle(clientID, 'BCS', simx_opmode_blocking)
    # res, goal = simxGetObjectHandle(clientID, 'Goal', simx_opmode_blocking)
    # S1 = np.zeros(6, dtype='int32')
    # for i in range(0, 6):
    #     res, S1[i] = simxGetObjectHandle(clientID, 'Tip' + str(i + 1), simx_opmode_blocking)
    # Tip_target = np.zeros(6, dtype='int32')
    # for i in range(0, 6):
    #     res, Tip_target[i] = simxGetObjectHandle(clientID, 'TipTarget' + str(i + 1), simx_opmode_blocking)

    # Lz = np.zeros(n+1)
    # init_position = np.zeros((6, 3))

    # # print(simxGetObjectPosition)
    # for i in range(6):
    #     res, init_position[i] = simxGetObjectPosition(clientID, S1[i], BCS, simx_opmode_oneshot_wait)
    # for i in range(1,n+1):
    #     Lz[i] = init_position[0][2] - i * 0.1/n

    # for i in range(1,n+1):
    #     for j in range(0,6,2):
    #         simxSynchronousTrigger(clientID)
    #         simxSetObjectPosition(clientID, Tip_target[j], BCS, [init_position[j][0], init_position[j][1], Lz[i]],
    #                      simx_opmode_oneshot_wait)
    # # simxSynchronousTrigger(clientID)
    # for i in range(1,n+1):
    #     for j in range(1,6,2):
    #         simxSynchronousTrigger(clientID)
    #         simxSetObjectPosition(clientID, Tip_target[j], BCS, [init_position[j][0], init_position[j][1], Lz[i]],
    #                            simx_opmode_oneshot_wait)
    # simxSynchronousTrigger(clientID)





    
