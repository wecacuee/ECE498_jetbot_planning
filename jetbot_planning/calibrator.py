from enum import Enum

import rclpy
from rclpy.node import Node

from aruco_opencv_msgs.msg import ArucoDetection
from geometry_msgs.msg import Twist
import numpy as np


VMIN = 0.12 # 13 cm per sec
OMIN = np.pi # one rotation per two sec
DT = 0.5
GOAL_MARKER_ID = 7

# https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py
def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    >>> numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    q = numpy.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle / 2.0) / qlen
    q[0] = math.cos(angle / 2.0)
    return q

def axangle_from_quat(quat):
    quat /= np.linalg.norm(quat, axis=-1)
    sin_theta_by_2 = np.linalg.norm(quat[..., 1:4], axis=-1)
    cos_theta_by_2 = quat[..., 0]
    theta = 2 * np.arctan2(sin_theta_by_2, cos_theta_by_2)
    axis = quat[..., 1:5] / sin_theta_by_2
    return axis, theta

def rotmat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def new_twist(linear_vel, ang_vel, scale_lin=-0.60, scale_ang=-0.70):
    twist = Twist()
    twist.linear.x = scale_lin * linear_vel
    twist.linear.y = 0.
    twist.linear.z = 0.
    twist.angular.x = 0.
    twist.angular.y = 0.
    twist.angular.z = scale_ang * ang_vel
    return twist

class MoveStopCycle(Enum):
    MOVE = 0
    STOP = 1

class Calibrator(Node):
    def __init__(self):
        super().__init__('calibrator')
        self.sub = self.create_subscription(ArucoDetection,
                                            '/aruco_detections', 
                                            self.on_aruco_detection, 10)
        self.pub = self.create_publisher(Twist, '/jetbot/cmd_vel', 10)
        self.counter = 5
        self.ignore_aruco_detection = False
        self.publish_zero_timer = None
        self.renable_subscriptions_timer = None
        self.marker2pose = {}
        self.prev_robot_pose_wrt_goal = []
        self.rate = self.create_rate(int(1.0/DT))

        self.timer = self.create_timer(DT, self.timer_callback)
        self.next_vel_to_pub = (0., 0.)
        self.robot_behaviour_state = {
            # The robot is lost. Rotate in place until marker is found
            'lost': False,
            # How much have we rotated since we were last lost
            'rotated_since_lost': 0,
            'no_robot_pose_counter': 0,

            # The robot published a move command or a stop command.
            # We are going to cycle between the two for consistency
            'move_stop_cycle': MoveStopCycle.STOP,

            # Are we ready for next cmd
            'ready_for_next_twist': False
        }

    def on_aruco_detection(self, msg):
        if not self.robot_behaviour_state['ready_for_next_twist']:
            return
        for m in msg.markers:
            axis, angle = axangle_from_quat(np.array(
                [m.pose.orientation.w, 
                 m.pose.orientation.x,
                 m.pose.orientation.y,
                 m.pose.orientation.z]))
            self.marker2pose[m.marker_id] = np.array(
                 [m.pose.position.x, m.pose.position.z,
                 angle])
        pose_goal = self.marker2pose[GOAL_MARKER_ID]
        robot_pose_wrt_goal = np.hstack((
            - rotmat(-pose_goal[2]) @ pose_goal[:2],
            - pose_goal[2]))
        #self.get_logger().info('Pose: {}'.format(robot_pose_wrt_goal))
        if len(self.prev_robot_pose_wrt_goal):
            self.get_logger().info('Avg Lin vel: {}'.format(
                sum([np.linalg.norm(prev[:2] - cur[:2]) / DT
                     for prev, cur in zip(self.prev_robot_pose_wrt_goal, 
                                          self.prev_robot_pose_wrt_goal[1:] +
                                          [robot_pose_wrt_goal])]) /
                len(self.prev_robot_pose_wrt_goal))
            )

        self.prev_robot_pose_wrt_goal.append(robot_pose_wrt_goal)
        if len(self.prev_robot_pose_wrt_goal) <= 5:
            self.next_vel_to_pub = (VMIN, 0.)
            self.robot_behaviour_state['ready_for_next_twist'] = False
        else:
            self.next_vel_to_pub = (0., 0.)
            self.robot_behaviour_state['ready_for_next_twist'] = False


    def _publish_move_stop(self, lin_ang_vel):
        moved = False
        # Alternate between move and stop cycles
        if self.robot_behaviour_state['move_stop_cycle'] == MoveStopCycle.MOVE:
            self.get_logger().info('Move {}'.format(lin_ang_vel))
            self.pub.publish(new_twist(*lin_ang_vel))
            moved = True
            self.robot_behaviour_state['move_stop_cycle'] = MoveStopCycle.STOP
        elif self.robot_behaviour_state['move_stop_cycle'] == MoveStopCycle.STOP:
            self.get_logger().info('Stop')
            self.pub.publish(new_twist(0., 0.))
            moved = False
            self.robot_behaviour_state['move_stop_cycle'] = MoveStopCycle.MOVE
        else:
            assert False, 'Only two possible values for move_stop_cycle'
        return moved


    def timer_callback(self):
        if self.robot_behaviour_state['lost']:
            self.get_logger().info('I am lost')
            if self.robot_behaviour_state['rotated_since_lost'] < 2*np.pi:
                # if the robot is lost, it is not see 
                if self._publish_move_stop((0., OMIN)):
                    self.robot_behaviour_state['robot_behaviour_state'] + OMIN*DT
            else:
                # we have rotated enough, we give up
                self.get_logger().warning('Rotated one full rotation. Giving up ')
                self._publish_move_stop((0., 0.))
        else:
            # The robot is not lost, we know where to go
            moved = self._publish_move_stop(self.next_vel_to_pub)
            if moved:
                self.next_twist_to_pub = new_twist(0., 0.)
                self.get_logger().info('ready_for_next_twist')
                self.robot_behaviour_state['ready_for_next_twist'] = True





def main(args=None):
    rclpy.init(args=args)
    astar = Calibrator()

    rclpy.spin(astar)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    astar.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
