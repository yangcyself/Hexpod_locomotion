import sys
sys.path.append("../")
import math
import random
import pickle
import blueGait as TYSenv
import actorcritic.oldenv as  kinetic
import vrep


class Robot:
    def __init__(self,clientID):
        self.pos = TYSenv.robot_position()
        self.target_pos = TYSenv.target_position()
        self.target_theta = -TYSenv.target_orient()

        self.D_max = 2.1  # the critical line of the safe region and unsafe region.
        self.SR = 0
        self.radius = 0.4

        errorCodeKinectDepth, kinectDepth = vrep.simxGetObjectHandle(clientID, 'kinect_depth',
                                                                     vrep.simx_opmode_oneshot_wait)
        self.obs_dis,self.obs_theta,closest_y_angle = kinetic.fetchKinect(kinectDepth)
        self.obs_theta = -self.obs_theta

        self.R = 0
        self.A = 0
        self.E = 0
        self.state = (0, 0, 0)

        self.Q = {}
        self.N = {}

        self.beta = 1
        self.gamma = 1

    def __update(self):
        self.pos = TYSenv.robot_position()
        self.target_pos = TYSenv.target_position()
        self.target_theta = -TYSenv.target_orient()
        errorCodeKinectDepth, kinectDepth = vrep.simxGetObjectHandle(clientID, 'kinect_depth',
                                                                     vrep.simx_opmode_oneshot_wait)
        self.obs_dis, self.obs_theta, closest_y_angle = kinetic.fetchKinect(kinectDepth)
        self.obs_theta = -self.obs_theta

    def __get_tar_dis(self):
        dx = self.pos[0] - self.target_pos[0]
        dy = self.pos[1] - self.target_pos[1]
        return (dx**2+dy**2)**(1/2)

    def __is_win_state(self):
        if self.__get_tar_dis() < self.radius:
            return True
        return False

    def __is_fail_state(self):
        if self.obs_dis < self.radius:
            return True
        return False

    def __get_state_R(self):
        if self.obs_dis <= self.D_max / 3:
            self.R = 0
            self.SR = 0
        elif self.obs_dis <= self.D_max * 2 / 3:
            self.R = 1
            self.SR = 0
        elif self.obs_dis <= self.D_max:
            self.R = 2
            self.SR = 0
        else:
            self.R = 3    # safe region
            self.SR = 1

    def __get_state_A(self):
        res = math.pi / 10
        for i in range(10):
            if i * res > (math.pi / 2 - self.obs_theta):
                self.A = i
                break

    def __get_state_E(self):
        res = math.pi / 4
        if self.target_theta>0:
            for i in range(4):
                if (i + 1) * res >= self.target_theta:
                    self.E = i
                    break
        else:
            for i in range(4):
                if -(i + 1) * res <= self.target_theta:
                    self.E = 7 - i
                    break

    def __get_state(self):
        self.__get_state_R()
        self.__get_state_A()
        self.__get_state_E()
        self.state = (self.R,self.A,self.E)

    def __exp_function(self,idx):
        self.__get_state()
        return self.Q[self.state][idx] + 0.1/self.N[self.state][idx]

    def init_Q(self):
        for r in range(4):
            for a in range(10):
                for e in range(8):
                    self.Q[(r, a, e)] = [0] * 3
                    self.N[(r, a, e)] = [1] * 3

    def __reward_function(self):
        self.__get_state()
        win = self.__is_win_state()
        fail = self.__is_fail_state()
        if win:
            return 2
        elif fail:
            return -1
        elif self.SR == 1:
            return 1
        else:
            return 0.05 * self.obs_dis + 0.2 * self.target_theta + 0.1 * self.obs_theta

    def __policy_run(self):
        while True:
            self.__get_state()
            current_state = self.state
            current_tar_theta = self.target_theta

            if self.SR == 1:
                print("Safe Region pos",self.pos[:2])
                if current_tar_theta > 0.5:
                    TYSenv.turnleft()
                    TYSenv.forward()
                elif -0.5 <= current_tar_theta <= 0.5:
                    TYSenv.forward()
                else:
                    TYSenv.turnright()
                    TYSenv.forward()
                self.__update()

            else:
                print("Unsafe Region pos",self.pos[:2])
                act_idx_value = []
                for i in range(3):
                    act_idx_value.append(self.__exp_function(i))
                act_idx = act_idx_value.index(max(act_idx_value))
                if act_idx == 0:
                
                    TYSenv.forward()
                elif act_idx == 1:
                    TYSenv.turnleft()
                else:
                    TYSenv.turnright()

                self.Q[current_state][act_idx] += self.beta*(self.__reward_function()+self.gamma*max(self.Q[self.state])-self.Q[current_state][act_idx])
                self.__update()
                if self.__is_fail_state():
                    break
            if self.__is_win_state():
                break


    def train(self,scenario=100):
        sce_num = scenario
        for i in range(sce_num):
            self.__policy_run()
            print("scenario:",i)
            TYSenv.reset()

    def run(self):
        self.__get_state()
        self.__update()
        current_state = self.state
        current_tar_theta = self.target_theta
        print(self.target_theta,self.target_pos)
        # print(TYSenv.robot_position(),TYSenv.target_orient())
        if self.SR == 1:
            if current_tar_theta > 0.2:
                print("leftforward")
                TYSenv.turnleft()
                TYSenv.forward()
            elif -0.2 <= current_tar_theta <= 0.2:
                print("forward")
                TYSenv.forward()
            else:
                print("rightforward")
                TYSenv.turnright()
                TYSenv.forward()

        else:
            Q_line = self.Q[current_state]
            act_idx = Q_line.index(max(Q_line))
            if act_idx == 0:
                TYSenv.forward()
            elif act_idx == 1:
                TYSenv.turnleft()
            else:
                TYSenv.turnright()
        return self.__is_win_state()

    def save_q_table(self):
        outfile = open('qtable.pickle','wb')
        pickle.dump(self.Q,outfile)
        outfile.close()
        outfile = open('ntable.pickle','wb')
        pickle.dump(self.N,outfile)
        outfile.close()

    def read_q_table(self):
        infile = open('qtable.pickle','rb')
        self.Q = pickle.load(infile)
        infile.close()
        infile = open("ntable.pickle",'rb')
        self.N = pickle.load(infile)
        infile.close()


if __name__ == "__main__":
    clientID = TYSenv.clientID
    robot1 = Robot(clientID)
    robot1.init_Q()
    kinetic.clientID = TYSenv.clientID
    kinetic.set_map()
    robot1.train(100)
    robot1.save_q_table()
    # robot1.read_q_table()
    # robot1.run()
    # robot1.run()
    # while(not robot1.run()):
    #     pass
    print(kinetic.target)
