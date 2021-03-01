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
        self.state = state
        self.f = 0
        self.g = 0
        self.h = 0
        self.past_actions = past_actions


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

    def generate_adj_states(self, cur_state):
        if cur_state[2] == 0:
            three_childs = []
            #if cur_state[0] <= int(self.world_width*self.resolution)-1:
                #three_childs.append([cur_state[0]+1, cur_state[1], cur_state[2]])
            three_childs.append((cur_state[0], cur_state[1], 1.0))
            three_childs.append((cur_state[0], cur_state[1], 3.0))
        elif cur_state[2] == 1:
            three_childs = []
            #if cur_state[1] <= int(self.world_height*self.resolution)-1:
                #three_childs.append([cur_state[0], cur_state[1]+1, cur_state[2]])
            three_childs.append((cur_state[0], cur_state[1], 2.0))
            three_childs.append((cur_state[0], cur_state[1], 0.0))
        elif cur_state[2] == 2:
            three_childs = []
            #if cur_state[0] >= 1:
                #three_childs.append([cur_state[0]-1, cur_state[1], cur_state[2]])
            three_childs.append((cur_state[0], cur_state[1], 3.0))
            three_childs.append((cur_state[0], cur_state[1], 1.0))
        else:
            three_childs = []
            #if cur_state[1] >= 1:
                #three_childs.append([cur_state[0], cur_state[1]-1, cur_state[2]])
            three_childs.append((cur_state[0], cur_state[1], 0.0))
            three_childs.append((cur_state[0], cur_state[1], 2.0))
        return three_childs

    def compare_state(self,state1,state2):
        if abs(state1[0]-state2[0])+abs(state1[1]-state2[1])+abs(state1[2]-state2[2]) == 0:
            return True
        else:
            return False

    def get_action_given_states(self, before, after):
        if after[2] == before[2]:
            return [1,0]
        elif (after[2]-before[2])==1 or after[2]==0 and before[2]==3:
            return [0,1]
        else:
            return [0,-1]

    def generate_plan(self):
        """TODO: FILL ME! This function generates the plan for the robot, given a goal.
        You should store the list of actions into self.action_seq.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """

        start_node = Node(parent = None, state = (1.0,1.0,0.0), past_actions = list())
        open_list = []
        open_list_set = set()
        closed_list = set()
        state_traj = []
        heapq.heappush(open_list,(start_node.f,start_node))
        open_list_set.add(start_node.state)
        while len(open_list) > 0:

            
            min_f_node = heapq.heappop(open_list)[1]
            open_list_set.remove(min_f_node.state)
            closed_list.add(min_f_node.state)
            #print(min_f_node.state)

            #if self.goal.pose.position.x == min_f_node.state[0] and self.goal.pose.position.y == min_f_node.state[1]:
            if self._check_goal(min_f_node.state):
                self.action_seq = min_f_node.past_actions
                print("STATE TRAJECTORY LENGTH: ", len(self.action_seq))
                print(self.action_seq)
                break

            adj_nodes = []
            for action in [[1,0],[0,1],[0,-1]]:
                move = self.discrete_motion_predict(min_f_node.state[0],min_f_node.state[1],min_f_node.state[2],action[0],action[1])
                if move:
                    past_act_list = copy.deepcopy(min_f_node.past_actions)
                    past_act_list.append(action)
                    adj_nodes.append(Node(parent = min_f_node , state = move, past_actions= past_act_list))

            for node in adj_nodes:
                bool_add = True
                if node.state in closed_list:
                    bool_add = False
                    continue
                node.g = min_f_node.g + 1
                node.h = abs(node.state[0] - self.goal.pose.position.x) + abs(node.state[1] - self.goal.pose.position.y) #manhattan distance without considering direction
                node.f = node.g + node.h
                if node.state in open_list_set:
                    bool_add = False
                    continue
                if bool_add:
                    heapq.heappush(open_list,(node.f,node))
                    open_list_set.add(node.state)


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

    def get_current_discrete_state(self):
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
            print("Sending actions:", action[0], action[1]*np.pi/2)
            msg = create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
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
    if planner.goal is not None:
        cur_time = time.time()
        planner.generate_plan()
        print("Time taken: ",time.time()-cur_time)

    # You could replace this with other control publishers
    planner.publish_discrete_control()

    # save your action sequence
    result = np.array(planner.action_seq)
    np.savetxt("actions_continuous.txt", result, fmt="%.2e")

    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')

    # spin the ros
    rospy.spin()
