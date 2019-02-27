# -^- coding:utf-8 -^-
import sys
sys.path.append("../")
from config import *
if (ENVIRONMENT=="BLUE"):
    from blueGait import *
else:
    from powerGait import *
import time
import numpy as np
from logger import Tlogger
tlogger = Tlogger
print(tlogger)

import matplotlib.pyplot as plt
SIDE = 0
PAIN_GAMMA = 1


ORIPOS=np.array([[ 5.27530670e-01 , 3.04633737e-01,-5.4652e-01],
        [-2.28881836e-05 , 6.09106421e-01,-5.4652e-01],
        [-5.27527809e-01 , 3.04500699e-01,-5.4652e-01],
        [ 5.27622223e-01 ,-3.04508090e-01,-5.4652e-01],
        [ 1.44958496e-04 ,-6.09171569e-01,-5.4652e-01],
        [-5.27413368e-01 ,-3.04654479e-01,-5.4652e-01]])
# ORIPOS = ORIPOS.dot(np.array([[1.05,0,0],
#                                 [0,1.05,0],
#                                 [0,0,1]]))


class observation_space:
    shape = (15,)

class action_space:
    shape = (6,)
    high = 0.1
topolist = []


def generateTarget():
    loc = np.random.rand(2)
    loc = (loc-0.5)*10
    loc = np.append(loc ,[0.2])
    loc = list(loc)
    # loc = np.array([5,0,0.67])
    return loc


def topograph(X,Y):
    if(MAP == "fence"):
        for x,y,r,h in topolist:
            if(abs(X-x)<r):
                return h
        return 0
    height = 0
    for x,y,r,h in topolist:
        if((x-X)**2+(y-Y)**2<= r**2):
            height = max(height,h)
    return height

def barrier_collision(X,Y,R):
    for x,y,r,h in topolist:
        if((x-X)**2+(y-Y)**2<= (r+R)**2):
            return 1
    return 0


def generate_set_TOPO_util(obj,r,h):
    global topolist
    if(MAP=="fence"):
        for f in obj:
            loc = generateTarget()
            vrep.simxSetObjectPosition(clientID, f, -1, loc,
                               vrep.simx_opmode_oneshot_wait)
            topolist.append((loc[0],loc[1],r,h))
        return
    for b in obj:
        loc = generateTarget()
        while(loc[0]**2+loc[1]**2<1 or barrier_collision(loc[0],loc[1],r)):
            loc = generateTarget()
        loc[2] = h/2
        vrep.simxSetObjectPosition(clientID, b, -1, loc,
                               vrep.simx_opmode_oneshot_wait)
        topolist.append((loc[0],loc[1],r,h))

def refresh_TOPO_util(obj,r,h):
    global topolist
    for b in obj:
        res,loc = vrep.simxGetObjectPosition(clientID, b, -1,
                               vrep.simx_opmode_oneshot_wait)
        topolist.append((loc[0],loc[1],r,h))

def refresh_TOPO():
    global topolist
    topolist = []
    if(MAP=="fence"):
        refresh_TOPO_util(Fence,0.05,0.1)
        return
    refresh_TOPO_util(Barrier,0.05,0.1)
    refresh_TOPO_util(Wall,0.25,0.5)

def generate_set_TOPO():
    if(not OBSERVETOPO):
        return
    global topolist
    topolist = []
    if(MAP=="fence"):
        generate_set_TOPO_util(Fence,0.05,0.1)
        return
    generate_set_TOPO_util(Barrier,0.05,0.1)
    generate_set_TOPO_util(Wall,0.25,0.5)

def set_map_util(obj,r,h,pos):
    for b,loc in zip(obj, pos):
        vrep.simxSetObjectPosition(clientID, b, -1, loc,vrep.simx_opmode_oneshot_wait)
        topolist.append((loc[0],loc[1],r,h))

def set_map():
    if(not OBSERVETOPO):
        return 
    set_map_util(Barrier,0.05,0.1,[[1,0,0.05]]*12)
    set_map_util(Wall,0.25,0.5,[[2,0,0.25]]*6)


def topoObservation():

    if(REFRESHTOPO):
        refresh_TOPO()

    resolution = 0.05
    scale = 1 #40*40  比 cifar10 还大
    X = np.arange(-scale,scale,resolution)
    Y = np.arange(-scale,scale,resolution)
    X, Y = np.meshgrid(X, Y)
    res, loc = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    location = loc[:2]
    res, loc = vrep.simxGetObjectOrientation (clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    ori = loc[2]
    #turn_neg_ori
    gX = X*np.cos(-ori) - Y*np.sin(-ori)
    gY = X*np.sin(-ori) + Y*np.cos(-ori)
    gX += location[0]
    gY += location[1]
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j] = topograph(gX[i][j],gY[i][j])
    return Z


def futherTopoObservation():
    if(REFRESHTOPO):
        refresh_TOPO()

    resolution = 0.4
    scale = 2.4 #40*40  比 cifar10 还大
    X = np.arange(-scale,scale+resolution,resolution)
    Y = np.arange(-scale,scale+resolution,resolution)
    X, Y = np.meshgrid(X, Y)
    res, loc = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    location = loc[:2]
    res, loc = vrep.simxGetObjectOrientation (clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    ori = loc[2]
    #turn_neg_ori
    gX = X*np.cos(-ori) - Y*np.sin(-ori)
    gY = X*np.sin(-ori) + Y*np.cos(-ori)
    gX += location[0]
    gY += location[1]
    Z = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if( 4<=i<9 and 4<=j<9):
               continue
            Z.append(topograph(gX[i][j],gY[i][j]))

    return list(np.array(Z).reshape(-1,))

def distance(obs):
    dst = 0
    for i in range(2):
        # dst += (obs[i+18]-obs[i+24])**2 
        dst += obs[i+12]**2
    return math.sqrt(dst)

def reset():
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
    time.sleep(5)
    status = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    if(not BLUEROBOT):
        recover(n=30)
    # print("status",status)

    global target
    global lastdist
    global bestdist

    if(SETTEDTOPO):
        refresh_TOPO()
    elif SETMAP:
        set_map()
        target = [4,0,0.2]
        vrep.simxSetObjectPosition(clientID, goal, -1, target,
                               vrep.simx_opmode_oneshot_wait)
    else:
        generate_set_TOPO()

        target = generateTarget()
        while(target[0]**2+target[1]**2<4):
            target = generateTarget()

        target = list(target)
        
        vrep.simxSetObjectPosition(clientID, goal, -1, target,
                               vrep.simx_opmode_oneshot_wait)

    if(RWD_PASS_WALL):
        initialize_wall_vec()
    obs = []
    # if(BLUEROBOT):
    #     vrep.updateRobotPosition()
    for i in range(6):
        res, loc = vrep.simxGetObjectPosition(clientID,S1[i],BCS,vrep.simx_opmode_oneshot_wait)
        loc = list(loc[:-1])
        obs+=loc
    # res, loc = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    # loc = list(loc)
    # obs+=loc
    # res, loc = vrep.simxGetObjectOrientation (clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    # loc = list(loc)
    # obs+=loc
    res , difftarget = vrep.simxGetObjectPosition (clientID,goal,BCS,vrep.simx_opmode_oneshot_wait)
    # obs+=list(difftarget)
    obs+=list(difftarget[:-1])
    
    obs.append(SIDE)
    dst = distance(obs)
    lastdist = dst
    bestdist = dst
    assert(len(obs)==observation_space.shape[0])
    if (FUTHERTOPO):
        obs+=futherTopoObservation()

    if(OBSERVETOPO):
        return (obs,topoObservation())

    return obs

def exit():
    vrep.simxFinish(clientID)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

rewardItems = []
def legPainful(obs):
    rwd = 0
    for i in range(6):
        pain  = obs[2*i+0]**2+obs[2*i+1]**2 - 0.5
        if(pain>0):
            rwd -= pain
    return rwd
rewardItems.append((legPainful,RWD_PAIN,RWDFAC_PAIN,"pain"))

def unit_vec(vec):
    return vec/np.sqrt(np.sum(vec**2))


# def dangerous(obs):
#     threshold = 0.6
#     rwd = 0
#     X,Y = obs[12],obs[13]
#     for x,y,r,h in topolist:
#         if(h<0.3):
#             continue
#         danger = min(0,math.sqrt((x-X)**2+(y-Y)**2)-threshold)
#         rwd -= danger**2
    # return rwd
# rewardItems.append((dangerous,RWD_DANEROUS,RWDFAC_DANEROUS,"danger"))

target_direction = None
wall_init_vec = []
"""
wall_init_vec consists of tuples, the x,y position of the wall ,and the reward for passing the wall
"""
def initialize_wall_vec():
    global wall_init_vec,target,target_direction

    _,robot_pos = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    robot_pos = np.array(robot_pos[:2])
    target_pos = np.array(target[:2])
    target_direction = unit_vec(target_pos - robot_pos)
    for x,y,r,h in topolist:
        if(r<0.1):
            continue
        wall_pos = np.array([x,y])
        robot_vec =  robot_pos - wall_pos
        target_vec = target_pos - wall_pos
        dot_prod = robot_vec.dot(target_vec)
        if(dot_prod < - r**2 and unit_vec(robot_vec).dot(unit_vec(target_vec)) < -0.5 ):# r**2 代表一定在连线上，乘个2扩大一点
            wall_init_vec.append((wall_pos,1))

def pass_wall_reward( _ ):
    global wall_init_vec
    wall_init = []
    rwd = 0
    _,robot_pos = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    robot_pos = np.array(robot_pos[:2])
    for wall_pos,r in wall_init_vec:
        robot_vec =  robot_pos - wall_pos
        if(robot_vec.dot(target_direction)>0):
            rwd += r
            print("passed one obj")
        else:
            wall_init.append((wall_pos,r))

    wall_init_vec = wall_init
    return rwd
rewardItems.append((pass_wall_reward,RWD_PASS_WALL,RWDFAC_PASS_WALL,"pass_wall"))


def display():
    for x,y,r,h in topolist:
        X=[x-r,x-r,x+r,x+r,x-r]
        Y=[y-r,y+r,y+r,y-r,y-r]
        ax.plot(X,Y)
    x = target[0]
    y = target[1]
    r = 0.2
    X=[x-r,x-r,x+r,x+r,x-r]
    Y=[y-r,y+r,y+r,y-r,y-r]
    ax.plot(X,Y)
    res, loc = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    x = loc[0]
    y = loc[1]
    r = 0.3
    X=[x-r,x-r,x+r,x+r,x-r]
    Y=[y-r,y+r,y+r,y-r,y-r]
    ax.plot(X,Y)


def step(action):

    global lastdist
    global bestdist
    global SIDE

    # global dist_ckp
    reward = 0
    done = False
    assert (lastdist>0)
    assert (len(action) ==action_space.shape[0])
    
    oriPos = ORIPOS[[a for a in range(SIDE,6,2)]]
    action = action.reshape(3,2)
    newaction = np.concatenate((action,np.zeros((3,1))),axis = 1)
    assert(newaction.shape==(3,3))
    if(NOISE):
        newaction+=np.random.normal(np.zeros_like(newaction),0.01)

    three_step(oriPos+newaction,SIDE)

    # three_step(np.zeros((3,3)),0)
    SIDE = 1-SIDE
    obs = []

    # if(BLUEROBOT):
    #     vrep.updateRobotPosition()
    for i in range(6):
        res, loc = vrep.simxGetObjectPosition(clientID,S1[i],BCS,vrep.simx_opmode_oneshot_wait)       
        loc = list(loc[:-1])
        obs+=loc
    res, loc = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    if(loc[2]<0.05 or np.isnan(loc[2]) or abs(loc[2])>1e+10):
        reward =-20
        print("@@@@@@ E X P L O D E @@@@@@")
        done = True
   
    res , difftarget = vrep.simxGetObjectPosition (clientID,goal,BCS,vrep.simx_opmode_oneshot_wait)
    obs+=list(difftarget[:-1])
    # obs+=list(difftarget)
    print("DIFFTARGET:",difftarget)

    obs.append(SIDE)
    assert(len(obs)==observation_space.shape[0])
    dst = distance(obs)
    # print(dst , bestdist)
    # print("orientation:", obs[-5])
    if(bestdist > dst and not done):
        reward += 2*(bestdist - dst)
        bestdist = dst
    if(not done):
        reward += min(0,lastdist -  dst)
    lastdist = dst
    if(dst < 0.5):
        reward = 40
        done = True
 
    tlogger.dist["rewardFunc"] = tlogger.dist.get("rewardFunc",0)+reward
    if(not done):
        for item, flag,fac,nam in rewardItems:
            if(flag):
                r = fac * item(obs)
                reward += r
                tlogger.dist[nam] = tlogger.dist.get(nam, 0) + r
    # print(reward)

    info = None

    if (FUTHERTOPO):
        obs+=futherTopoObservation()
    if(NOISE):
        obs = np.array(obs)
        obs+= np.random.normal([0]*len(obs),0.01)
        obs = list(obs)

    if(DISPLAY_OBS ):
        # topoobs = topoObservation()
        ax.clear()
        display()
        # for i in range(6):
        #     topoobs[int((obs[i]+1)*20)][int((obs[i+1]+1)*20)] = 0.05
        # ax.imshow(topoobs)
        # ax.autoscale([-5,5],[-5,5])
        
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        fig.canvas.draw()

    if(OBSERVETOPO):
        return (obs,topoObservation()) ,reward, done, info

    return obs ,reward, done, info


running  = False
target = generateTarget()
while(target[0]**2+target[1]**2<4):
    target = generateTarget()
target = [4,0,0.2]
vrep.simxSetObjectPosition(clientID, goal, -1, target,
                               vrep.simx_opmode_oneshot_wait)
lastdist = -1
bestdist = -1

def mystep(act):
    # act = act.reshape(3,2)
    three_step(act,0)

if(DISPLAY_OBS ):
    fig = plt.figure()
    ax = fig.gca()
    # obs = topoObservation()
    # ax.imshow(obs)
    display()
    # ax.autoscale([-5,5],[-5,5])
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    # fig.canvas.draw()
    plt.show(block=False) 

"""
interface for tcd easier q learning
"""
def fetchKinect(kinect_depth):
    # res, loc = vrep.simxGetObjectPosition(clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    # location = loc[:2]
    # res, loc = vrep.simxGetObjectOrientation (clientID,BCS,-1,vrep.simx_opmode_oneshot_wait)
    # ori = loc[2]
    shortestD = 100
    shortestA = 0
    for b in Wall:
        _,loc = vrep.simxGetObjectPosition(clientID,b,BCS,vrep.simx_opmode_oneshot_wait)
        dis = math.sqrt(loc[0]**2+loc[1]**2) - 0.25
        if(dis<shortestD):
            shortestD = dis
            shortestA = math.atan2(loc[1],loc[0])
        print("shortestD:",shortestD)
    return shortestD,shortestA,0



if __name__ == "__main__":
    # reset()
    # mystep(np.zeros((3,3)))
    # step(np.zeros(6))
    print(topoObservation())