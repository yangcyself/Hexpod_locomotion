# -^- coding:utf-8 -^-
import bluerep as vrep
N = 1

import numpy as np
import math
import time
"""
可以调用的接口：three_step_delta(newpos_delta,side)
                three_step(newpos,side)
            传入的参数都是一个3*3的tensor，一个side取0或1代表哪组腿。
            只是delta是原来腿的位置基础上的差值
            均是以body的位置作为坐标系确定的点
            机器人的姿态永远有两个约束，body的位置在足尖的中心点，六个腿没有扭动（角度的平均值是0）
用于测试和使用的更高级接口: walk_a_step(length=0.6,deg=0):
                        turn_a_deg(deg):
辅助接口：
        print_steps()画出所有足尖相对身体的位置
"""

# N=75

def recover(n=N):
    Lz = np.zeros(n+1)
    init_position = np.zeros((6, 3))
    time.sleep(3)
    # vrep.updateRobotPosition()
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


def bodyDiffOri(target):
    ang = np.zeros(6)
    basAng = [1,3,5,-1,-3,-5]
    basAng = np.array(basAng)*math.pi/6
    for i in range(0,6):
        ang[i] = math.atan2(target[i][1],target[i][0])
    deg = ave_ang(ang-basAng)
    return deg



def three_step_delta(newpos,side,MOD="delta"):
    """
    newpos is the difference between the new position and the present position of the three peds
    3*3, the z
    side = 0 or 1
    assume that the body position is above the middle of the foot (x,y)s.
    """
    initPos = np.zeros((6, 3))
    # vrep.updateRobotPosition()
    for i in range(6):
        res, initPos[i] = vrep.simxGetObjectPosition(clientID, S1[i], BCS, vrep.simx_opmode_oneshot_wait)
    # height = 0.25
    if(MOD=="delta"):
        newpos_delta = newpos
    else:
        newpos_delta = np.zeros((3,3))
        for i in range(side,6,2):
            newpos_delta[int(i/2)] = newpos[int(i/2)] - initPos[i]
    newpos_delta = np.clip(newpos_delta,-0.1,0.1)
    avedelta = np.sum(newpos_delta,axis=0)/6 
    target = initPos-avedelta

    pee = []
    for i in range(6):
        if(i%2==side):
            target[i] += newpos_delta[int(i/2)]
            pee += list(newpos_delta[int(i/2)][:2])
    # # transTo(target)
    # print(pee)
    
    # print(target)
    peb = list(avedelta    )+[0,0,bodyDiffOri(target)] #经验之举，否则转反了 #但是应该代表需要转的角度
    peb[2]=0
    # pee = np.clip(pee,-0.1,0.1)
    vrep.robotSetFoot(side,pee,peb)
    time.sleep(2)

    
def three_step(newpos,side,initPos=None):
    three_step_delta(newpos,side,MOD="abslute")

def print_steps():
    import matplotlib.pyplot as plt
    x=[]
    y = []
    # vrep.updateRobotPosition()
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


    # Tip 末梢 坐标系
    S1 = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, S1[i] = vrep.simxGetObjectHandle(clientID, 'Tip' + str(i + 1), vrep.simx_opmode_blocking)
    # 坐标系，与 Tip上边连接在一起
    Tip_target = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, Tip_target[i] = vrep.simxGetObjectHandle(clientID, 'TipTarget' + str(i + 1), vrep.simx_opmode_blocking)



"""
interface for tcd qlearning
"""
def robot_position(): 
    return vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)[1][:2]
def target_position():
    return vrep.simxGetObjectPosition(clientID,goal,-1,vrep.simx_opmode_oneshot_wait)[1][:2]
def target_orient():
    _,loc = vrep.simxGetObjectPosition(clientID,goal,BCS,vrep.simx_opmode_oneshot_wait)
    print("loc:",loc)
    return math.atan2(loc[1],loc[0])
def turnleft():
    turn_a_deg(0.2)
def turnright():
    turn_a_deg(-0.2)
def forward():
    walk_a_step(0.1,0)
def reset():
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    time.sleep(2)
    status = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

def exit():
    vrep.simxFinish(clientID)
    time.sleep(1)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)


if __name__=="__main__":
    #test
    # recover( n=min(N,30))
    walk_a_step(0.1,0)
    turn_a_deg(0.2)
    # three_step(np.array([[-0.01539664 ,-0.07759521  ,0.        ],
    #                 [-0.09819948 ,-0.05909955  ,0.        ],
    #                 [ 0.10317233 ,-0.05715271  ,0.        ]]),0)
    