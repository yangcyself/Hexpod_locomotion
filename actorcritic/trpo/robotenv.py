import sys
sys.path.append("../")

from sixlegged_4 import  *
import vrep
import time
import numpy as np


def generateTarget():
    target = np.random.rand(2)
    target = (target-0.5)*10
    target = np.append(target ,[0.67])
    target = list(target)
    target = [-5,5,0.67]
    return target

def distance(obs):
    dst = 0
    # for i in range(3):
    #     dst += (obs[i+18]-obs[i+24])**2 
    dst = obs[24]**2+obs[25]**2
    return math.sqrt(dst)

def reset():
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    time.sleep(2)
    status = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    print("status",status)
    global target
    global lastdist
    global bestdist
    # global dist_ckp
    target = generateTarget()
    while(target[0]**2+target[1]**2<4):
        target = generateTarget()
    target = list(target)
    # dist_ckp = [False]*5
    obs = []
    for i in range(6):
        res, loc = vrep.simxGetObjectPosition(clientID,Sf[i],body,vrep.simx_opmode_oneshot_wait)
        obs+=loc
    for i in range(6):
        res, loc = vrep.simxGetObjectPosition(clientID,Sf[i],-1,vrep.simx_opmode_oneshot_wait)
        obs.append(loc[2])

    # res, loc = vrep.simxGetObjectPosition(clientID,body,-1,vrep.simx_opmode_oneshot_wait)
    # obs+=loc
    # res, loc = vrep.simxGetObjectOrientation (clientID,body,-1,vrep.simx_opmode_oneshot_wait)
    # obs+=loc
    
    res, loc = vrep.simxGetObjectPosition (clientID,goal,body,vrep.simx_opmode_oneshot_wait)
    obs+=loc
    # difftarget = [target[i]-obs[i+18] for i in range(2)]
    # difftarget.append(target[2])
    # obs+=target
    # obs+=difftarget

    dst = distance(obs)
    lastdist = dst
    bestdist = dst
    # if(dst < 2):
    #     dist_ckp[1] = True
    # elif(dst < 3):
    #         dist_ckp[2] = True
    # elif(dst < 4):
    #         dist_ckp[3] = True
    # elif(dst < 4.5):
    #         dist_ckp[4] = True
    # print(len(obs))
    assert(len(obs)==27)
    return obs

def exit():
    vrep.simxFinish(clientID)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

def step(action):
    global lastdist
    global bestdist
    # global dist_ckp
    reward = 0
    done = False
    assert (lastdist>0)
    assert (len(action) ==18)
    for i,l in enumerate(action):
        # l = l/10+0.05
        vrep.simxSetJointTargetPosition(clientID, pole[i+1], l, vrep.simx_opmode_oneshot)
    time.sleep(0.7)
    obs = []
    for i in range(6):
        res, loc = vrep.simxGetObjectPosition(clientID,Sf[i],body,vrep.simx_opmode_oneshot_wait)
        obs+=loc
    for i in range(6):
        res, loc = vrep.simxGetObjectPosition(clientID,Sf[i],-1,vrep.simx_opmode_oneshot_wait)
        obs.append(loc[2])

    res, loc = vrep.simxGetObjectPosition(clientID,body,-1,vrep.simx_opmode_oneshot_wait)
    if(loc[2]<0.45):
        reward -=20
        done = True
    print(loc, end = " ")

    res, loc = vrep.simxGetObjectPosition (clientID,goal,body,vrep.simx_opmode_oneshot_wait)
    obs+=loc

    # obs+=loc
    # res, loc = vrep.simxGetObjectOrientation (clientID,body,-1,vrep.simx_opmode_oneshot_wait)
    # obs+=loc
    # reward -= (math.sqrt(loc[0]**2 + loc[0]**2))/10
    # difftarget = [target[i]-obs[i+18] for i in range(2)]
    # difftarget.append(target[2])
    # # obs+=target
    # obs+=difftarget
    # print(difftarget,end = " ")
    assert(len(obs)==27)
    dst = distance(obs)
    print(dst , bestdist)
    if(bestdist > dst):
        reward += 4*(bestdist - dst)
        bestdist = dst
    reward += min(0,lastdist -  dst)
    lastdist = dst
    if(dst < 0.5):
        reward  =20
        done = True

    print(reward)

    info = None
    return obs ,reward, done, info


running  = False
# target = generateTarget()
# while(target[0]**2+target[1]**2<4):
#     target = generateTarget()
# lastdist = -1
# bestdist = -1
# dist_ckp = [False]*5



print('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to V-REP, set a very large time-out for blocking commands
if clientID!=-1:
    print ('Connected to remote API server')
    emptyBuff = bytearray()
    # Start the simulation:
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    #Retrive the body
    res, body = vrep.simxGetObjectHandle(clientID, 'body', vrep.simx_opmode_blocking)
    res, goal = vrep.simxGetObjectHandle(clientID, 'Target', vrep.simx_opmode_blocking)
    # Retrieve the poles
    pole = np.zeros(19, dtype='int32')
    for i in range(1, 19):
        res, pole[i] = vrep.simxGetObjectHandle(clientID, 'P' + str(i), vrep.simx_opmode_blocking)
    # Retrive the U1
    U1 = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, U1[i] = vrep.simxGetObjectHandle(clientID, 'Hip' + str(i + 1), vrep.simx_opmode_blocking)
    # Retrive the U2
    U2 = np.zeros(6, dtype='int32')
    res, U2[0] = vrep.simxGetObjectHandle(clientID, 'shaft2', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, U2[i] = vrep.simxGetObjectHandle(clientID, 'shaft2#' + str(i - 1), vrep.simx_opmode_blocking)
    # Retrive the U3
    U3 = np.zeros(6, dtype='int32')
    res, U3[0] = vrep.simxGetObjectHandle(clientID, 'shaft3', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, U3[i] = vrep.simxGetObjectHandle(clientID, 'shaft3#' + str(i - 1), vrep.simx_opmode_blocking)
    # Retrive the target Dummy
    target = np.zeros(6, dtype='int32')
    res, target[0] = vrep.simxGetObjectHandle(clientID, 'target_Dummy', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, target[i] = vrep.simxGetObjectHandle(clientID, 'target_Dummy#' + str(i - 1), vrep.simx_opmode_blocking)

    #Retrive the support points
    Dummy = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, Dummy[i] = vrep.simxGetObjectHandle(clientID, 'Sup_Dummy' + str(i), vrep.simx_opmode_blocking)

    # Retrive the S1
    S1 = np.zeros(6, dtype='int32')
    for i in range(0, 6):
        res, S1[i] = vrep.simxGetObjectHandle(clientID, 'TipSf' + str(i + 1), vrep.simx_opmode_blocking)
    # Retrive the S2
    S2 = np.zeros(6, dtype='int32')
    res, S2[0] = vrep.simxGetObjectHandle(clientID, 'S2', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, S2[i] = vrep.simxGetObjectHandle(clientID, 'S2#' + str(i - 1), vrep.simx_opmode_blocking)
    # Retrive the S3
    S3 = np.zeros(6, dtype='int32')
    res, S3[0] = vrep.simxGetObjectHandle(clientID, 'S3', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, S3[i] = vrep.simxGetObjectHandle(clientID, 'S3#' + str(i - 1), vrep.simx_opmode_blocking)
    #Retrive the Spheres
    Sf = np.zeros(6, dtype='int32')
    res, Sf[0] = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_blocking)
    for i in range(1, 6):
        res, Sf[i] = vrep.simxGetObjectHandle(clientID, 'Sphere#' + str(i - 1), vrep.simx_opmode_blocking)
    