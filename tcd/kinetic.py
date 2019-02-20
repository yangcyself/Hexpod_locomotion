import vrep
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Depth2Cloud(depthSTR):
        errorCodeKinectDepth,kinectDepth=vrep.simxGetObjectHandle(clientID,depthSTR,vrep.simx_opmode_oneshot_wait)
        errorHere,resol,depth=vrep.simxGetVisionSensorDepthBuffer(clientID,kinectDepth,vrep.simx_opmode_oneshot_wait)

        camXAngleInDegrees = 57
        camXResolution = 640
        camYResolution = 480
        camXHalfAngle = camXAngleInDegrees*0.5*math.pi/180
        camYHalfAngle = (camXAngleInDegrees*0.5*math.pi/180)*camYResolution/camXResolution
        nearClippingPlane = 0.01
        depthAmplitude = 3.5
        closest_dis = 100
        closest_x_angle = 0
        closest_y_angle = 0

        #calculate the inner parameters
        f_x = -camXResolution /(2 * depthAmplitude * math.tan(camXAngleInDegrees*math.pi / 180))
        f_y = -camXResolution /(2 * depthAmplitude * math.tan(camXAngleInDegrees*math.pi / 180 *camYResolution / camXResolution))
        # print(f_x)
        print(len(depth))
        u0 = camXResolution / 2
        v0 = camYResolution / 2

        cloud_graph = []
        for i in range(camXResolution):
            for j in range(camYResolution):
                tmp = []
                depth_value = depth[i+(j)*camXResolution]

                zCoor = nearClippingPlane + depthAmplitude*depth_value
                xCoor = (i - u0)*zCoor / f_x
                yCoor = (j - v0)*zCoor / f_y

                tmp.append(xCoor)
                tmp.append(yCoor)
                tmp.append(zCoor)
                cloud_graph.append(tmp)

        return cloud_graph

def fetchKinect(kinectDepth):


        errorHere,resol,depth=vrep.simxGetVisionSensorDepthBuffer(clientID,kinectDepth,vrep.simx_opmode_oneshot_wait)
        # plt.imshow(np.array(depth).reshape([480,640]), cmap = "gray")
        # plt.show()
        camXAngleInDegrees = 57
        camXResolution = 640
        camYResolution = 480
        camXHalfAngle = camXAngleInDegrees*0.5*math.pi/180
        camYHalfAngle = (camXAngleInDegrees*0.5*math.pi/180)*camYResolution/camXResolution
        nearClippingPlane = 0.01
        depthAmplitude = 3.5
        closest_dis = 100
        closest_x_angle = 0
        closest_y_angle = 0

        for i in range(camXResolution):
            Xangle = (camXResolution/2 - i - 0.5)/camXResolution * camXHalfAngle * 2
            for j in range(225,camYResolution):
                Yangle = (j - camYResolution/2 + 0.5)/camYResolution * camYHalfAngle * 2

                depth_value = depth[i+(j)*camXResolution]
                zCoor = nearClippingPlane + depthAmplitude*depth_value
                xCoor = math.tan(Xangle)*zCoor
                yCoor = math.tan(Yangle)*zCoor

                dist = math.sqrt(zCoor*zCoor + xCoor*xCoor + yCoor*yCoor)
                if(dist < closest_dis):
                    closest_dis = dist
                    closest_x_angle = Xangle
                    closest_y_angle = Yangle
                    p = i
                    q = j

        return closest_dis,closest_x_angle,closest_y_angle


clientID = 0
#
# if __name__ == "__main__":
#     print('Program started')
#     vrep.simxFinish(-1) # just in case, close all opened connections
#     clientID=vrep.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to V-REP, set a very large time-out for blocking commands
#     if clientID!=-1:
#         print ('Connected to remote API server')
#
#         emptyBuff = bytearray()
#
#         # Start the simulation:
#         vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
#
#         errorCodeKinectDepth,kinectDepth=vrep.simxGetObjectHandle(clientID,'kinect_depth',vrep.simx_opmode_oneshot_wait)
#     for i in range(100):
#         closest_dis,closest_x_angle,closest_y_angle = fetchKinect(kinectDepth)
#         print(closest_dis)



# if __name__ == "__main__":

#         global depthArr

#         #Connection
#         vrep.simxFinish(-1)
#         clientID=vrep.simxStart('127.0.0.1',19997,True,True,-5000000,5)

#         if clientID != -1:
#                 print("Connected to remote API Server")

#         else:
#                 print("Connection not succesfull")
#                 sys.exit("Could not connect")


#         for i in range(100):
#             closest_dis,closest_x_angle,closest_y_angle = fetchKinect(kinectDepth)
#             print(closest_dis)
        # a = Depth2Cloud('kinect_depth')

        # fig = plt.figure()
        # # 创建3d图形的两种方式
        # # ax = Axes3D(fig)
        # ax = fig.add_subplot(111, projection='3d')
        # # X, Y value
        # X = np.arange(-4, 4, 0.25)
        # Y = np.arange(-4, 4, 0.25)
        # X = []
        # Y = []
        # Z = []
        # for item in a:
        #     X.append(item[0])
        #     Y.append(item[1])
        #     Z.append(item[2])
        # X = np.array(X)
        # Y = np.array(Y)
        # Z = np.array(Z)
        # ax.plot(X, Y, Z)

        # plt.show()

        # print(closest_dis,closest_x_angle,closest_y_angle)




