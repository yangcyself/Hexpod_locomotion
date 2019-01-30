# -^- coding:utf-8 -^-
from actorcritic.config import *
if ENVIRONMENT=="VREP":
    import vrep
    N = 50
else:
    import toyrep as vrep
    N=5
import numpy as np
import math
import time
import copy
"""
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

# N=75

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
            vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, [init_position[j][0], init_position[j][1], Lz[i]],
                               vrep.simx_opmode_oneshot_wait)
    for i in range(1,n+1):
        vrep.simxSynchronousTrigger(clientID)
        for j in range(1,6,2):
            vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, [init_position[j][0], init_position[j][1], Lz[i]],
                               vrep.simx_opmode_oneshot_wait)

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



def transTo(target,n=N): #TODO: make max step length or 

    assert(target.shape==(6,3))
    initPos = np.zeros((6, 3))
    for i in range(6):
        res, initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    delta = (target - initPos)/n
#     print (delta)
    #To make the steps smoother
    for i in range(3):
        initPos += delta/3
        vrep.simxSynchronousTrigger(clientID)
        for j in range(6):
            vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, initPos[j],vrep.simx_opmode_oneshot_wait)

    for i in range(n-2):
        initPos += delta
        vrep.simxSynchronousTrigger(clientID)
        for j in range(6):
            vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, initPos[j],vrep.simx_opmode_oneshot_wait)


    #To make the steps smoother
    for i in range(3):
        initPos += delta/3
        vrep.simxSynchronousTrigger(clientID)
        for j in range(6):
            vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, initPos[j],vrep.simx_opmode_oneshot_wait)
    for j in range(6):
        vrep.simxSetObjectPosition(clientID, Tip_target[j], BCS, target[j],vrep.simx_opmode_oneshot_wait)
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
    height = 0.25
    avedelta = np.sum(newpos_delta,axis=0)/6 

    #lift up:
    initPos = np.zeros((6, 3))
    for i in range(6):
        res, initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    target = initPos
    for i in range(6):
        if(i%2==side):
            target[i][2] +=height
    transTo(target)

    target = initPos-avedelta
    for i in range(6):
        if(i%2==side):
            target[i] += newpos_delta[int(i/2)]
    target = detork(target)
    transTo(target)
    for i in range(6):
        if(i%2==side):
            target[i][2] -=height
    target_sum = np.sum(target,axis=0)/6
    target_sum[2] = 0
    target-=target_sum
    target = detork(target)
    transTo(target)
    # avedelta = np.sum(newpos_delta,axis=0)/6 

    #lift up:
    # initPos = np.zeros((6, 3))
    # for i in range(6):
    #     res, initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)

    # target = initPos-avedelta/2
    # for i in range(6):
    #     if(i%2==side):
    #         target[i] += newpos_delta[int(i/2)]/2
    #         target[i][2] +=0.3

    # target = detork(target)
    # transTo(target)
    # target-=avedelta/2
    # for i in range(6):
    #     if(i%2==side):
    #         target[i] += newpos_delta[int(i/2)]/2
    #         target[i][2] -=0.3
    # target_sum = np.sum(target,axis=0)/6
    # target_sum[2] = 0
    # target-=target_sum
    # target = detork(target)
    # transTo(target)

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

def turn_a_deg(deg):
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
    res = vrep.simxSynchronous(clientID, True)

    emptyBuff = bytearray()

    # Start the simulation:
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    # ???
    res, U4 = vrep.simxGetObjectHandle(clientID, 'J11R', vrep.simx_opmode_blocking)
    res, U5 = vrep.simxGetObjectHandle(clientID, 'shaft19', vrep.simx_opmode_blocking)
    # 摄像机
    res, kinect_depth_camera = vrep.simxGetObjectHandle(clientID, 'kinect_depth', vrep.simx_opmode_blocking)
    res, kinect_rgb_camera = vrep.simxGetObjectHandle(clientID, 'kinect_rgb', vrep.simx_opmode_blocking)
    res, kinect_joint = vrep.simxGetObjectHandle(clientID, 'kinect_joint', vrep.simx_opmode_blocking)
    #Retrive the body coordinate system
    # 坐标系 ????
    res, GCS = vrep.simxGetObjectHandle(clientID, 'GCS', vrep.simx_opmode_blocking)
    res, BCS = vrep.simxGetObjectHandle(clientID, 'BCS', vrep.simx_opmode_blocking)
    res, goal = vrep.simxGetObjectHandle(clientID, 'Goal', vrep.simx_opmode_blocking)
    # Retrive the poles
    # 18个腿上的杆
    pole = np.zeros(19, dtype='int32')
    for i in range(1, 19):
        res, pole[i] = vrep.simxGetObjectHandle(clientID, 'P' + str(i), vrep.simx_opmode_blocking)

    Barrier = np.zeros(10, dtype='int32')
    for i in range(0, 10):
        res, Barrier[i] = vrep.simxGetObjectHandle(clientID, 'Barrier' + str(i), vrep.simx_opmode_blocking)

    Wall = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, Wall[i] = vrep.simxGetObjectHandle(clientID, 'Wall' + str(i), vrep.simx_opmode_blocking)

    #Retrive the U1
    # 在腿和主题连接处。？？？
    U1 = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, U1[i] = vrep.simxGetObjectHandle(clientID, 'Hip' + str(i + 1), vrep.simx_opmode_blocking)
    # Retrive the U2
    # 每根腿。最上层
    U2 = np.zeros(6, dtype='int32')
    res, U2[0] = vrep.simxGetObjectHandle(clientID, 'J21R', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, U2[i] = vrep.simxGetObjectHandle(clientID, 'J21R' + str(i - 1), vrep.simx_opmode_blocking)
    # Retrive the U3
    # 每根腿，最上层
    U3 = np.zeros(6, dtype='int32')
    res, U3[0] = vrep.simxGetObjectHandle(clientID, 'J31R', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, U3[i] = vrep.simxGetObjectHandle(clientID, 'J31R' + str(i - 1), vrep.simx_opmode_blocking)
    # Retrive the target Dummy
    # ？？？？？
    target_Dummy = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, target_Dummy[i] = vrep.simxGetObjectHandle(clientID, 'target_Dummy' + str(i + 1),
                                                        vrep.simx_opmode_blocking)

    # Retrive the S1
    # Tip 末梢 坐标系
    S1 = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, S1[i] = vrep.simxGetObjectHandle(clientID, 'Tip' + str(i + 1), vrep.simx_opmode_blocking)
    # 坐标系，与 Tip上边连接在一起
    Tip_target = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, Tip_target[i] = vrep.simxGetObjectHandle(clientID, 'TipTarget' + str(i + 1), vrep.simx_opmode_blocking)


if __name__=="__main__":
    #test
    recover( n=min(N,30))
    walk_a_step(0.3,math.pi/2)
    turn_a_deg(0.5)
    # three_step(np.array([[-0.01539664 ,-0.07759521  ,0.        ],
    #                 [-0.09819948 ,-0.05909955  ,0.        ],
    #                 [ 0.10317233 ,-0.05715271  ,0.        ]]),0)
    