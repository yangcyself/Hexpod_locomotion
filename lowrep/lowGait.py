# -^- coding:utf-8 -^-
"""
使用方法：
env: from lowGait import *

说明:
该文件是利用low level controller 提供的反解功能，通过控制杆长与vrep 交互
应该打开model3ttt

核心内容 closeLoopSetPos:
    调用神经网络获取杆长把腿移到目标位置上，为了确保效果，会在腿到了目标位置之前一直重复此操作

可以调用的接口：three_step_delta(newpos_delta,side)
            three_step(newpos,side)
            传入的参数都是一个3*3的tensor，一个side取0或1代表哪组腿。
            只是delta是原来腿的位置基础上的差值
            均是以body的位置作为坐标系确定的点
            机器人的姿态永远有两个约束，body的位置在足尖的中心点，六个腿没有扭动（角度的平均值是0）
用于测试和使用的更高级接口: walk_a_step(length=0.6,deg=0):
                        turn_a_deg(deg):
辅助接口：reset(),
        print_steps()画出所有足尖相对身体的位置


"""
# -^- coding:utf-8 -^-
# from actorcritic.config import *
# if ENVIRONMENT=="VREP":
#     import vrep
#     N = 50
# else:
#     import toyrep as vrep
#     N=5
import sys
sys.path.append("../")
import vrep
N=2
import numpy as np
import math
import time
import copy
# from FootControl  import *
import LLController as llc



accept_threshold = 0.001

class converter:
    def __init__(self):
        pass


def recover(n=N):
    Lz = np.zeros(n+1)
    init_position = np.zeros((6, 3))
    time.sleep(3)
    for i in range(6):
        res, init_position[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    for i in range(1,n+1):
        Lz[i] = init_position[0][2] - i * 0.1/n

    for i in range(1,n+1):
        vrep.simxSynchronousTrigger(clientID)
        for j in range(0,6,2):
            init_position[j][2] = Lz[i]
        closeLoopSetPos(init_position)
    for i in range(1,n+1):
        vrep.simxSynchronousTrigger(clientID)
        for j in range(1,6,2):
            init_position[j][2] = Lz[i]
        closeLoopSetPos(init_position)

# def reset():
#     vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
#     time.sleep(2)
#     status = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
#     recover()

def ave_ang(angs):
    angs = np.array(angs)
    x = np.cos(angs)
    y = np.sin(angs)
    return math.atan2(np.sum(y),np.sum(x))
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


def getleglength(n):
    res = []
    for j in range(3):
        _,l = vrep.simxGetJointPosition(clientID, pole[3*n+j+1], vrep.simx_opmode_oneshot_wait)
        res.append(l)
    return np.array(res)

def setleglength(n,l):
    # targetdistance = 
    # accept_threshold
    for j in range(3):
        vrep.simxSetJointTargetPosition(clientID, pole[3*n+j+1], l[j], vrep.simx_opmode_oneshot)

totalerror = 0
errorcount = 0

def closeLoopSetPos(target):
    assert(target.shape == (6,3))
    threshold = 0.012
    initPos = np.zeros((6, 3))
    for i in range(6):
        _,initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    delta = target - initPos
    z_delta = np.copy(delta)
    z_delta[abs(delta) < threshold] = 0
    adjustCount = 0
    while(np.sum(abs(z_delta))!=0):
        adjustCount +=1
        if(adjustCount >3):
            break
        # print(delta)
        for i in range(6):
            if(np.sum(abs(z_delta[i]))==0):
                continue
            leglength=getleglength(i)
            pos = llc.querry(np.concatenate((leglength,delta[i])),i)
            # if(i==0):
                # print(pos)
            setleglength(i,leglength + pos) # I just don't know why it should be minus
        time.sleep(0.1)
        for i in range(6):
            _,initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
        delta = target - initPos
        z_delta = np.copy(delta)
        z_delta[abs(delta) < threshold] = 0
    error = np.sqrt( np.sum(delta[0]**2))
    global totalerror, errorcount
    totalerror += error
    errorcount += 1
    print(totalerror/errorcount)

    # print("Finished")


def transTo(target,n=N): #TODO: make max step length or 

    assert(target.shape==(6,3))
    initPos = np.zeros((6, 3))
    for i in range(6):
        res, initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    delta = (target - initPos)/n
#     print (delta)
    for i in range(n):
        initPos += delta
        vrep.simxSynchronousTrigger(clientID)
        closeLoopSetPos(initPos)
            # vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, pos,vrep.simx_opmode_oneshot_wait)
        time.sleep(0.1)
    #To make the steps smoother
    closeLoopSetPos(initPos)
        # vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, pos,vrep.simx_opmode_oneshot_wait)
    vrep.simxSynchronousTrigger(clientID)

def detork(target):
    """
    make the position have no zhuan dong
    """
    # dist = np.zeros(6)
    ang = np.zeros(6)
    basAng = [1,3,5,-1,-3,-5]
    basAng = np.array(basAng)*math.pi/6
    for i in range(0,6):
        # dist[i] = math.sqrt(target[i][0]**2+target[i][1]**2)
        ang[i] = math.atan2(target[i][1],target[i][0])
        
    # print(ang)
    # print(np.sum(ang)/6,-ave_ang(ang))
    # ang-=np.sum(ang)/6
    deg = ave_ang(ang-basAng)
    # ang -= deg
    # deg = -ave_ang(ang)
    # target_ = copy.deepcopy(target)
    for i in range(6):        
        # target[i][0] = dist[i]*math.cos(ang[i])
        # target[i][1] = dist[i]*math.sin(ang[i])
        # print("target_")
        # print(target_[i])
        target[i] = turnVec(target[i],-deg)
        # print("target")
        # print(target[i])
    return target

def three_step_delta(newpos_delta,side):
    """
    newpos is the difference between the new position and the present position of the three peds
    3*3, the z
    side = 0 or 1
    assume that the body position is above the middle of the foot (x,y)s.
    """
    height = 0.1
    avedelta = np.sum(newpos_delta,axis=0)/6 

    #lift up:
    initPos = np.zeros((6, 3))
    for i in range(6):
        res, initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    target = initPos
    for i in range(6):
        if(i%2==side):
            target[i][2] +=height
    time.sleep(1)
    transTo(target)

    target = initPos-avedelta
    for i in range(6):
        if(i%2==side):
            target[i] += newpos_delta[int(i/2)]
    target = detork(target)
    time.sleep(1)
    transTo(target)
    for i in range(6):
        if(i%2==side):
            target[i][2] -=height
    target_sum = np.sum(target,axis=0)/6
    target_sum[2] = 0
    target-=target_sum
    target = detork(target)
    time.sleep(1)
    transTo(target)

def three_step(newpos,side):

    initPos = np.zeros((3, 3))
    for i in range(side,6,2):
        res,initPos[int(i/2)]=vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    three_step_delta(newpos-initPos,side)

def print_steps():
    import matplotlib.pyplot as plt
    x=[]
    y = []
    for i in range(6):
        res, Pos = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
        x.append(Pos[0])
        y.append(Pos[1])
    plt.plot(x,y,"d")
    plt.show()


def walk_a_step(length=0.3,deg=0):
    x = length*math.cos(deg)
    y = length*math.sin(deg)
    three_step_delta(np.array([[x,y,0] for i in range(3)]),0)
    three_step_delta(np.array([[x,y,0] for i in range(3)]),1)

def turn_a_deg(deg = 0.2):
    target = np.zeros((3,3))
    for i in range(0,6,2):
        res,target[int(i/2)]=vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    for i in range(3):
        dist = math.sqrt(target[i][0]**2+target[i][1]**2)
        ang = math.atan2(target[i][1],target[i][0])
        target[i][0] = dist*math.cos(ang+deg)
        target[i][1] = dist*math.sin(ang+deg)
    three_step(target,0)
    
    for i in range(1,6,2):
        res,target[int(i/2)]=vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    for i in range(3):
        dist = math.sqrt(target[i][0]**2+target[i][1]**2)
        ang = math.atan2(target[i][1],target[i][0])
        target[i][0] = dist*math.cos(ang+deg)
        target[i][1] = dist*math.sin(ang+deg)
    three_step(target,1)



print('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to V-REP, set a very large time-out for blocking commands
if clientID!=-1:
    print ('Connected to remote API server')
    # res = vrep.simxSynchronous(clientID, True)

    # emptyBuff = bytearray()

    # Start the simulation:
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    res, BCS = vrep.simxGetObjectHandle(clientID, 'body', vrep.simx_opmode_blocking)
    
    # res, goal = vrep.simxGetObjectHandle(clientID, 'Goal', vrep.simx_opmode_blocking)
    # Retrive the poles
    # 18个腿上的杆
    pole = np.zeros(19, dtype='int32')
    for i in range(1, 19):
        res, pole[i] = vrep.simxGetObjectHandle(clientID, 'P' + str(i), vrep.simx_opmode_blocking)

    # Barrier = np.zeros(12, dtype='int32')
    # for i in range(0, 12):
    #     res, Barrier[i] = vrep.simxGetObjectHandle(clientID, 'Barrier' + str(i), vrep.simx_opmode_blocking)

    # Wall = np.zeros(6, dtype='int32')
    # for i in range(0, 6):
    #     res, Wall[i] = vrep.simxGetObjectHandle(clientID, 'Wall' + str(i), vrep.simx_opmode_blocking)

    # Fence = np.zeros(2, dtype='int32')
    # for i in range(0, 2):
    #     res, Fence[i] = vrep.simxGetObjectHandle(clientID, 'Fence' + str(i), vrep.simx_opmode_blocking)


    #Retrive the U1
    # 在腿和主题连接处。？？？
    U1 = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, U1[i] = vrep.simxGetObjectHandle(clientID, 'Hip' + str(i + 1), vrep.simx_opmode_blocking)
    # Retrive the U2
    # 每根腿。最上层
    U2 = np.zeros(6, dtype='int32')
    res, U2[0] = vrep.simxGetObjectHandle(clientID, 'shaft2', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, U2[i] = vrep.simxGetObjectHandle(clientID, 'shaft2#' + str(i - 1), vrep.simx_opmode_blocking)
    # Retrive the U3
    # 每根腿，最上层
    U3 = np.zeros(6, dtype='int32')
    res, U3[0] = vrep.simxGetObjectHandle(clientID, 'shaft3', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, U3[i] = vrep.simxGetObjectHandle(clientID, 'shaft3#' + str(i - 1), vrep.simx_opmode_blocking)







    # Retrive the target Dummy
    # ？？？？？
    # target_Dummy = np.zeros(6, dtype='int32')
    # for i in range(0, 6):
    #     res, target_Dummy[i] = vrep.simxGetObjectHandle(clientID, 'target_Dummy' + str(i + 1),
    #                                                     vrep.simx_opmode_blocking)

    # Retrive the S1
    # Tip 末梢 坐标系
    S1 = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, S1[i] = vrep.simxGetObjectHandle(clientID, 'TipSf' + str(i + 1), vrep.simx_opmode_blocking)

if __name__=="__main__":
    #test
    recover( n=min(N,30))
    walk_a_step(0.3,math.pi/2)
    turn_a_deg(0.5)
    # three_step(np.array([[-0.01539664 ,-0.07759521  ,0.        ],
    #                 [-0.09819948 ,-0.05909955  ,0.        ],
    #                 [ 0.10317233 ,-0.05715271  ,0.        ]]),0)
    