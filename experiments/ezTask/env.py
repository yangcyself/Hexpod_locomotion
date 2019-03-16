# -^- coding:utf-8 -^-
import sys
sys.path.append("../")
import time
import numpy as np
import matplotlib.pyplot as plt
import math


PAIN_GAMMA = 1
ORIPOS=np.array([[ 5.27530670e-01 , 3.04633737e-01],
        [-2.28881836e-05 , 6.09106421e-01],
        [-5.27527809e-01 , 3.04500699e-01],
        [ 5.27622223e-01 ,-3.04508090e-01],
        [ 1.44958496e-04 ,-6.09171569e-01],
        [-5.27413368e-01 ,-3.04654479e-01]])

FUTHERTOPO = True
OBSERVETOPO = True
DISPLAY_OBS = True
Barrier = 20
Wall = 10

class observation_space:
    shape = (2,)

class action_space:
    shape = (2,)
    high = 0.1


topolist = []

def generateTarget():
    loc = np.random.rand(2)
    loc = (loc-0.5)*10
    return loc


def topograph(X,Y):
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
    for b in range(obj):
        loc = generateTarget()
        while(loc[0]**2+loc[1]**2<1 or barrier_collision(loc[0],loc[1],r)):
            loc = generateTarget()
        topolist.append((loc[0],loc[1],r,h))

def generate_set_TOPO():
    if(not OBSERVETOPO):
        return
    global topolist
    topolist = []
    generate_set_TOPO_util(Barrier,0.05,0.1)
    generate_set_TOPO_util(Wall,0.25,0.5)

def topoObservation():
    global position
    resolution = 0.05
    scale = 1 #40*40  比 cifar10 还大
    X = np.arange(-scale,scale,resolution)
    Y = np.arange(-scale,scale,resolution)
    X, Y = np.meshgrid(X, Y)
    location = position
    ori = 0
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
    global position
    resolution = 0.4
    scale = 2.4 #40*40  比 cifar10 还大
    X = np.arange(-scale,scale+resolution,resolution)
    Y = np.arange(-scale,scale+resolution,resolution)
    X, Y = np.meshgrid(X, Y)
    location = position
    ori = 0
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
        dst += obs[i]**2  
    return math.sqrt(dst)

def reset():
    global target
    global lastdist
    global bestdist
    global position

    generate_set_TOPO()

    target = generateTarget()
    while(target[0]**2+target[1]**2<4):
        target = generateTarget()


    obs = []
    obs +=list(target-position)    
    dst = distance(obs)
    lastdist = dst
    bestdist = dst
    assert(len(obs)==observation_space.shape[0])
    if (FUTHERTOPO):
        obs+=futherTopoObservation()

    if(OBSERVETOPO):
        return (obs,topoObservation())
    return obs


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

    for i in ORIPOS:

        x = position[0] + i[0]
        y = position[1] + i[1]
        r = 0.1
        X=[x-r,x-r,x+r,x+r,x-r]
        Y=[y-r,y+r,y+r,y-r,y-r]
        ax.plot(X,Y)


def step(action):

    global lastdist
    global bestdist
    global position
    # global dist_ckp
    reward = 0
    done = False
    assert (lastdist>0)
    assert (len(action) ==action_space.shape[0])
    position = position + action
    obs = []

    obs = []
    obs +=list(target-position)    
    assert(len(obs)==observation_space.shape[0])

    for i in ORIPOS:
        legpos = position + i
        if(topograph(legpos[0],legpos[1])!=0):
            print("EXPLODE")
            done = True
            reward = -20

    dst = distance(obs)

    if(bestdist > dst and not done):
        reward += 2*(bestdist - dst)
        bestdist = dst
    if(not done):
        reward += min(0,lastdist -  dst)
    lastdist = dst
    if(dst < 0.5):
        reward = 20
        done = True
 
    info = None

    if (FUTHERTOPO):
        obs+=futherTopoObservation()
    if(DISPLAY_OBS ):
        ax.clear()
        display()        
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        fig.canvas.draw()

    if(OBSERVETOPO):
        return (obs,topoObservation()) ,reward, done, info

    return obs ,reward, done, info


running  = False
target = generateTarget()
position = np.zeros((2,))
while(target[0]**2+target[1]**2<4):
    target = generateTarget()

lastdist = -1
bestdist = -1

if(DISPLAY_OBS ):
    fig = plt.figure()
    ax = fig.gca()
    display()
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    plt.show(block=False) 


if __name__ == "__main__":
    print(topoObservation())