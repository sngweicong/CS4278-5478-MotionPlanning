#!/usr/bin/env python
import rospy
import numpy as np

from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *
from math import *
import copy
import argparse
import heapq
import time
import json

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit


def dump_action_table(action_table, filename):
    """dump the MDP policy into a json file

    Arguments:
        action_table {dict} -- your mdp action table. It should be of form {'1,2,0': (1, 0), ...}
        filename {str} -- output filename
    """
    tab = dict()
    for k, v in action_table.items():
        key = [str(i) for i in k]
        key = ','.join(key)
        tab[key] = v

    with open(filename, 'w') as fout:
        json.dump(tab, fout)

class Node():
    def __init__(self, parent = None, state = None, past_actions = []):
        self.parent = parent
        self.state= (state[0], state[1], self.zero_to_two_pi(state[2]))
        self.f = 0
        self.g = 0
        self.h = 0
        self.past_actions = past_actions
        self.xy_grid_size = 0.25
        self.theta_grid_size = pi/8
        self.discretized_state = (self.discrete_xy(state[0]), self.discrete_xy(state[1]), self.discrete_theta(self.state[2]))

    def zero_to_two_pi(self, theta):
        return theta - floor(theta/2/pi)*2*pi

    def discrete_xy(self, x_or_y):
        return floor(x_or_y/self.xy_grid_size)*self.xy_grid_size

    def discrete_theta(self, theta):
        return floor(theta/self.theta_grid_size)*self.theta_grid_size




class Planner:
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        """init function of the base planner. You should develop your own planner
        using this class as a base.

        For standard mazes, width = 200, height = 200, resolution = 0.05. 
        For COM1 map, width = 2500, height = 983, resolution = 0.02

        Arguments:
            world_width {int} -- width of map in terms of pixels
            world_height {int} -- height of map in terms of pixels
            world_resolution {float} -- resolution of map

        Keyword Arguments:
            inflation_ratio {int} -- [description] (default: {3})
        """
        rospy.init_node('planner')
        self.map = None
        self.pose = None
        self.goal = None
        self.path = None
        self.action_seq = None  # output
        self.aug_map = None  # occupancy grid with inflation
        self.action_table = {}

        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution
        self.inflation_ratio = inflation_ratio
        self.map_callback()
        self.sb_obs = rospy.Subscriber('/scan', LaserScan, self._obs_callback)
        self.sb_pose = rospy.Subscriber(
            '/base_pose_ground_truth', Odometry, self._pose_callback)
        self.sb_goal = rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped, self._goal_callback)
        self.controller = rospy.Publisher(
            '/mobile_base/commands/velocity', Twist, queue_size=10)
        rospy.sleep(1)



    def map_callback(self):
        """Get the occupancy grid and inflate the obstacle by some pixels. You should implement the obstacle inflation yourself to handle uncertainty.
        """
        self.map = rospy.wait_for_message('/map', OccupancyGrid).data
        #print(self.map[150*2500:151*2500])
        #print(self.map[60*200:61*200])

        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask
        #Sanity Checker, ignore
        '''
        self.world_width = 10
        self.world_height = 10
        self.fake_tuple = (-1,)*44 + (100,) + (-1,)*55
        self.fake_np = np.array(self.fake_tuple).reshape(self.world_width,self.world_height)
        self.pad = np.array([100]*((self.world_width+self.inflation_ratio*2)*(self.world_height+self.inflation_ratio*2))).reshape(self.world_width+self.inflation_ratio*2, self.world_height+self.inflation_ratio*2)
        self.pad[self.inflation_ratio:-self.inflation_ratio,self.inflation_ratio:-self.inflation_ratio] = self.fake_np
        self.aug_pad = copy.deepcopy(self.pad)
        for i in range(self.inflation_ratio,self.world_height+self.inflation_ratio):
            for j in range(self.inflation_ratio,self.world_width+self.inflation_ratio):
                if self.pad[i,j] == 100:
                    self.aug_pad[i-self.inflation_ratio:i+self.inflation_ratio+1,j-self.inflation_ratio:j+self.inflation_ratio+1] = 100
        self.aug_map = tuple(self.aug_pad[self.inflation_ratio:-self.inflation_ratio,self.inflation_ratio:-self.inflation_ratio].reshape(-1))
        print(self.aug_map)
        '''

        self.fake_np = np.array(self.map).reshape(self.world_height,self.world_width)
        self.pad = np.array([100]*((self.world_width+self.inflation_ratio*2)*(self.world_height+self.inflation_ratio*2))).reshape(self.world_height+self.inflation_ratio*2, self.world_width+self.inflation_ratio*2)
        self.pad[self.inflation_ratio:-self.inflation_ratio,self.inflation_ratio:-self.inflation_ratio] = self.fake_np
        self.aug_pad = copy.deepcopy(self.pad)
        for i in range(self.inflation_ratio,self.world_height+self.inflation_ratio):
            for j in range(self.inflation_ratio,self.world_width+self.inflation_ratio):
                if self.pad[i,j] == 100:
                    self.aug_pad[i-self.inflation_ratio:i+self.inflation_ratio+1,j-self.inflation_ratio:j+self.inflation_ratio+1] = 100
        self.aug_map = tuple(self.aug_pad[self.inflation_ratio:-self.inflation_ratio,self.inflation_ratio:-self.inflation_ratio].reshape(-1))


    def _pose_callback(self, msg):
        """get the raw pose of the robot from ROS

        Arguments:
            msg {Odometry} -- pose of the robot from ROS
        """
        self.pose = msg

    def _goal_callback(self, msg):
        self.goal = msg
        self.generate_plan()

    def _get_goal_position(self):
        goal_position = self.goal.pose.position
        return (goal_position.x, goal_position.y)

    def set_goal(self, x, y, theta=0):
        """set the goal of the planner

        Arguments:
            x {int} -- x of the goal
            y {int} -- y of the goal

        Keyword Arguments:
            theta {int} -- orientation of the goal; we don't consider it in our planner (default: {0})
        """
        a = PoseStamped()
        a.pose.position.x = x
        a.pose.position.y = y
        a.pose.orientation.z = theta
        self.goal = a

    def _obs_callback(self, msg):
        """get the observation from ROS; currently not used in our planner; researve for the next assignment

        Arguments:
            msg {LaserScan} -- LaserScan ROS msg for observations
        """
        self.last_obs = msg

    def _d_from_goal(self, pose):
        """compute the distance from current pose to the goal; only for goal checking

        Arguments:
            pose {list} -- robot pose

        Returns:
            float -- distance to the goal
        """
        goal = self._get_goal_position()
        return sqrt((pose[0] - goal[0])**2 + (pose[1] - goal[1])**2)

    def _check_goal(self, pose):
        """Simple goal checking criteria, which only requires the current position is less than 0.25 from the goal position. The orientation is ignored

        Arguments:
            pose {list} -- robot post

        Returns:
            bool -- goal or not
        """
        if self._d_from_goal(pose) < 0.25:
            return True
        else:
            return False

    def create_control_msg(self, x, y, z, ax, ay, az):
        """a wrapper to generate control message for the robot.

        Arguments:
            x {float} -- vx
            y {float} -- vy
            z {float} -- vz
            ax {float} -- angular vx
            ay {float} -- angular vy
            az {float} -- angular vz

        Returns:
            Twist -- control message
        """
        message = Twist()
        message.linear.x = x
        message.linear.y = y
        message.linear.z = z
        message.angular.x = ax
        message.angular.y = ay
        message.angular.z = az
        return message

    # def reward_fn(self, sprime_x, sprime_y, s_x, s_y): ##call for motion_checker twice
    #     check_n_grid = 5
    #     goal = self._get_goal_position()
    #     if sprime_x == goal[0] and sprime_y == goal[1]:
    #         return 1
    #     elif sprime_x == s_x and sprime_y == s_y:
    #         return 0
    #     else:
    #         x_grid = [(sprime_x-s_x)/5*i+s_x for i in range(1,6)]
    #         y_grid = [(sprime_y-s_y)/5*i+s_y for i in range(1,6)]
    #         for i in range(5):
    #             if self.collision_checker(x_grid[i],y_grid[i]):
    #                 return -10
    #         return 0

    def reward_function(self, s, true_a):
        #true_a is different from a.
        sprime = self.discrete_motion_predict(s[0], s[1], s[2], true_a[0], true_a[1])
        goal = self._get_goal_position()
        if sprime is None:
            return -50
        elif sprime[0] == goal[0] and sprime[1] == goal[1]:
            return 10
        else:
            return -1

    def ob_boundaries(self, coord, is_x):
        # pushes sprime to within (0,0) to (floor(width x res), floor(height x res)) if sprime go out of bounds.
        if is_x and coord > floor(self.world_width*self.resolution):
            coord = floor(self.world_width*self.resolution)
        elif is_x and coord < 0:
            coord = 0
        elif not is_x and coord > floor(self.world_height*self.resolution):
            coord = floor(self.world_height*self.resolution)
        elif not is_x and coord < 0:
            coord = 0
        return coord


    def prob_sprime_given_s_a(self,s,a): #use analytic formula to output 0.05
        #assumes no obstacles, collision checking is handled at reward function.
        #prevents checking outside of (0,0) to (floor(width x res), floor(height x res))
        # first value of tuple is probability
        # second is sprime
        # third is "true" action executed, different from a.
        if a == (0,0):
            return [(1, (s[0], s[1], s[2]), a)]

        elif a == (0,1) or a == (0,-1):
            if self.discrete_motion_predict(s[0], s[1], s[2], a[0], a[1]):
                return [(1, (s[0], s[1], (s[2]+a[1])%4), a)]
            else:
                return [(1, (s[0], s[1], s[2]), a)]
        else:
            new_s = [s[0],s[1]]
            if s[2] == 0 and s[0] < floor(self.world_width*self.resolution):
                new_s[0] += 1
            elif s[2] == 1 and s[1] < floor(self.world_height*self.resolution):
                new_s[1] += 1
            elif s[2] == 2 and s[0] > 0:
                new_s[0] -= 1
            elif s[2] == 3 and s[1] > 0:
                new_s[1] -= 1
            forward_left_x = self.ob_boundaries(round(s[0] - sin(s[2]*pi/2) + sin(s[2]*pi/2 + pi/2)), True)
            forward_left_y = self.ob_boundaries(round(s[1] + cos(s[2]*pi/2) - cos(s[2]*pi/2 + pi/2)), False)
            forward_right_x = self.ob_boundaries(round(s[0] + sin(s[2]*pi/2) - sin(s[2]*pi/2 - pi/2)), True)
            forward_right_y = self.ob_boundaries(round(s[1] - cos(s[2]*pi/2) + cos(s[2]*pi/2 - pi/2)), False)

            list_of_p_sprime_given_sa =[]
            if self.discrete_motion_predict(s[0], s[1], s[2], a[0], a[1]):
                list_of_p_sprime_given_sa.append((0.9, (new_s[0], new_s[1], s[2]), a))
            else:
                list_of_p_sprime_given_sa.append((0.9, (s[0], s[1], s[2]), a))

            if self.discrete_motion_predict(s[0], s[1], s[2], np.pi/2, 1):
                list_of_p_sprime_given_sa.append((0.05, (forward_left_x, forward_left_y, (s[2]+1)%4), (np.pi/2,1)))
            else:
                list_of_p_sprime_given_sa.append((0.05, (s[0], s[1], s[2]), (np.pi/2,1)))

            if self.discrete_motion_predict(s[0], s[1], s[2], np.pi/2, -1):
                list_of_p_sprime_given_sa.append((0.05, (forward_right_x, forward_right_y, (s[2]-1)%4), (np.pi/2,-1)))
            else:
                list_of_p_sprime_given_sa.append((0.05, (s[0], s[1], s[2]), (np.pi/2,-1)))

            return list_of_p_sprime_given_sa

    def debug_print(self):
        print(self.prob_sprime_given_s_a((10,10,0),(0,1)))
        print(self.prob_sprime_given_s_a((10,10,0),(0,-1)))
        print(self.prob_sprime_given_s_a((10,10,0),(1,0)))
        print('--------')
        print(self.discrete_motion_predict(1,1,1,np.pi/2,1))
        print(self.discrete_motion_predict(15,30,0,np.pi/2,-1))
        print(self.discrete_motion_predict(30,8,0,np.pi/2,1))
        print('--------')
        print(self.discrete_motion_predict(9,6,1,1,0))


    def generate_plan(self):
        """TODO: FILL ME! This function generates the plan for the robot, given a goal.
        You should store the list of actions into self.action_seq.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """
        if args.com:
            self.disc_rate = 0.9
        else:
            self.disc_rate = 0.99
        self.possible_actions = [(1,0),(0,1),(0,-1),(0,0)]
        max_delta_vs = 999         
        v_s = np.zeros([int(floor(self.world_width*self.resolution)+1),int(floor(self.world_height*self.resolution)+1),4])
        while (max_delta_vs > 0.001):
            print("start v iter loop",max_delta_vs)
            max_delta_vs = 0 
            older_v_s = copy.deepcopy(v_s)
            v_s = np.zeros([int(floor(self.world_width*self.resolution)+1),int(floor(self.world_height*self.resolution)+1),4])
            for x in range(int(floor(self.world_width*self.resolution)+1)):
                for y in range(int(floor(self.world_height*self.resolution)+1)):
                    for theta in range(4):
                        value_given_action = np.array([0.0,0.0,0.0,0.0])
                        for action_idx in range(len(self.possible_actions)):
                            prob_sprime_given_a = self.prob_sprime_given_s_a([x,y,theta],self.possible_actions[action_idx])
                            for each_prob_sprime in prob_sprime_given_a:
                                value_given_action[action_idx] += each_prob_sprime[0] * (self.reward_function((x,y,theta),each_prob_sprime[2]) + self.disc_rate * older_v_s[each_prob_sprime[1][0],each_prob_sprime[1][1],each_prob_sprime[1][2]])
                        #print(value_given_action)
                        v_s[x,y,theta] = np.max(value_given_action)
                        max_delta_vs = max(max_delta_vs, abs(older_v_s[x,y,theta]-v_s[x,y,theta]))
        #print("end v iter")
        #print(v_s.transpose(2,0,1))
        for x in range(int(floor(self.world_width*self.resolution)+1)):
            for y in range(int(floor(self.world_height*self.resolution)+1)):
                for theta in range(4):
                    value_given_action = np.array([0.0,0.0,0.0,0.0])
                    for action_idx in range(len(self.possible_actions)):
                        prob_sprime_given_a = self.prob_sprime_given_s_a([x,y,theta],self.possible_actions[action_idx])
                        for each_prob_sprime in prob_sprime_given_a:
                            value_given_action[action_idx] += each_prob_sprime[0] * (self.reward_function((x,y,theta),each_prob_sprime[2]) + self.disc_rate * v_s[each_prob_sprime[1][0],each_prob_sprime[1][1],each_prob_sprime[1][2]])  
                    self.action_table[x,y,theta] = self.possible_actions[np.argmax(value_given_action)]

        #print(self.action_table[goal[0],goal[1]-1,1])
        #print(self.action_table[goal[0],goal[1]-1,2])
        #print(self.action_table[goal[0],goal[1]-1,0])
        #print(self.action_table[goal[0],goal[1]+1,3])
        #print(self.action_table[goal[0],goal[1]+1,0])
        #print(self.action_table[goal[0],goal[1]+1,2])

    def get_current_continuous_state(self):
        """Our state is defined to be the tuple (x,y,theta). 
        x and y are directly extracted from the pose information. 
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot
        """
        x = self.pose.pose.pose.position.x
        y = self.pose.pose.pose.position.y
        orientation = self.pose.pose.pose.orientation
        ori = [orientation.x, orientation.y, orientation.z,
               orientation.w]

        phi = np.arctan2(2 * (ori[0] * ori[1] + ori[2] * ori[3]), 1 - 2 *
                         (ori[1] ** 2 + ori[2] ** 2))
        return (x, y, phi)

    def get_current_state(self):
        """Our state is defined to be the tuple (x,y,theta). 
        x and y are directly extracted from the pose information. 
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot in discrete space, e.g., (1, 1, 1) where the robot is facing north
        """
        x, y, phi = self.get_current_continuous_state()
        def rd(x): return int(round(x))
        return rd(x), rd(y), rd(phi / (np.pi / 2))

    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size
        
        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
        
        Returns:
            bool -- True for collision, False for non-collision
        """
        max_x = int(self.world_width * self.resolution - self.resolution)
        max_y = int(self.world_height * self.resolution - self.resolution)
        if x > max_x or x < 0 or y > max_y or y < 0:
            return True

        grid_x = int(x / self.resolution)-1
        grid_y = int(y / self.resolution)-1
        tuple_idx = grid_x + grid_y * self.world_width
        #print(tuple_idx)
        if self.aug_map[tuple_idx] == 100:
            return True
        else:
            return False

    def motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """predict the next pose of the robot given controls. Returns None if the robot collide with the wall
        The robot dynamics are provided in the homework description

        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
            theta {float} -- current theta of robot
            v {float} -- linear velocity 
            w {float} -- angular velocity

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        num_steps = int(dt * frequency)
        dx = 0
        dy = 0
        for i in range(num_steps):
            if w != 0:
                dx = - v / w * np.sin(theta) + v / w * \
                    np.sin(theta + w / frequency)
                dy = v / w * np.cos(theta) - v / w * \
                    np.cos(theta + w / frequency)
            else:
                dx = v*np.cos(theta)/frequency
                dy = v*np.sin(theta)/frequency
            x += dx
            y += dy
            #print(x,y)

            if self.collision_checker(x, y):
                return None
            theta += w / frequency
        return x, y, theta

    def discrete_motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """discrete version of the motion predict. Note that since the ROS simulation interval is set to be 0.5 sec
        and the robot has a limited angular speed, to achieve 90 degree turns, we have to execute two discrete actions
        consecutively. This function wraps the discrete motion predict.

        Please use it for your discrete planner.

        Arguments:
            x {int} -- current x of robot
            y {int} -- current y of robot
            theta {int} -- current theta of robot
            v {int} -- linear velocity
            w {int} -- angular velocity (0, 1, 2, 3)

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        w_radian = w * np.pi/2
        first_step = self.motion_predict(x, y, theta*np.pi/2, v, w_radian)
        if first_step:
            second_step = self.motion_predict(
                first_step[0], first_step[1], first_step[2], v, w_radian)
            if second_step:
                return (round(second_step[0]), round(second_step[1]), round(second_step[2] / (np.pi / 2)) % 4)
        return None

    def publish_control(self):
        """publish the continuous controls
        """
        for action in self.action_seq:
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1])
            self.controller.publish(msg)
            rospy.sleep(0.6)

    def publish_discrete_control(self):
        """publish the discrete controls
        """
        for action in self.action_seq:
            msg = self.create_control_msg(
                action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)

    def publish_stochastic_control(self):
        """publish stochastic controls in MDP. 
        In MDP, we simulate the stochastic dynamics of the robot as described in the assignment description.
        Please use this function to publish your controls in task 3, MDP. DO NOT CHANGE THE PARAMETERS :)
        We will test your policy using the same function.
        """
        current_state = self.get_current_state()
        actions = []
        new_state = current_state
        while not self._check_goal(current_state):
            current_state = self.get_current_state()
            action = self.action_table[current_state[0],
                                       current_state[1], current_state[2] % 4]
            if action == (1, 0):
                r = np.random.rand()
                if r < 0.9:
                    action = (1, 0)
                elif r < 0.95:
                    action = (np.pi/2, 1)
                else:
                    action = (np.pi/2, -1)
                action = (1, 0) #deterministic
            print("Sending actions:", action[0], action[1]*np.pi/2)
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            time.sleep(1)
            current_state = self.get_current_state()


if __name__ == "__main__":
    # TODO: You can run the code using the code below
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, default='1,8',
                        help='goal position')
    parser.add_argument('--com', type=int, default=0,
                        help="if the map is com1 map")
    args = parser.parse_args()

    try:
        goal = [int(pose) for pose in args.goal.split(',')]
    except:
        raise ValueError("Please enter correct goal format")

    if args.com:
        width = 2500
        height = 983
        resolution = 0.02
    else:
        width = 200
        height = 200
        resolution = 0.05

    # TODO: You should change this value accordingly
    if args.com:
        inflation_ratio = 10
    else:
        inflation_ratio = 7
    planner = Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    #planner.debug_print()
    if planner.goal is not None:
        cur_time = time.time()
        planner.generate_plan()
        print("Time taken: ",time.time()-cur_time)

    # You could replace this with other control publishers
    planner.publish_stochastic_control()

    # save your action sequence
    #result = np.array(planner.action_seq)
    #np.savetxt("actions_continuous.txt", result, fmt="%.2e")

    # for MDP, please dump your policy table into a json file
    dump_action_table(planner.action_table, 'mdp_policy.json')

    # spin the ros
    rospy.spin()
