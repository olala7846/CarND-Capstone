#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import tf.transformations
import angles

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add other member variables you need below
        self.base_lane = None
        self.traffic_wp = -1
        self.obstacle_wp = None
        self.seqnum = 0
        self.car_wp_q = -1
        self.target_velocity = 20.0
        self.last_traffic_wp_processed = -1

        rospy.spin()

    def pose_cb(self, msg):
        #rospy.loginfo("pose_cb timestamp %s x=%d y=%d z=%d", msg.header.stamp, msg.pose.position.x,
        #              msg.pose.position.y, msg.pose.position.z)
        if self.base_lane != None:
            quaternion = (msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w)
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
            veh_yaw = yaw % (2.0 * math.pi)
            veh_x = msg.pose.position.x
            veh_y = msg.pose.position.y
            
            # the course hits an inflection point at x=2339 yaw=90+ at which point x starts to decrease
            # next inflection at x=155 yaw=270+ at which point x starts to increase
            # in front of the car = increasing x for yaw between 270 and 90 and decreasing x from 90 to 270

            #veh_fwd = True
            #if veh_yaw > (math.pi/2) and veh_yaw <= (3*math.pi/2):  # 90 to 270 degrees
            #    veh_fwd = False

            pub_waypoints = []
            wp_start = -1
            min_dist = 1e9
            dist_q = 1e9
            state = 0  # no progress finding waypoint
            if len(self.base_lane.waypoints) > 1:
                # find the first waypoint in front of the vehicle,
                #   then take a sequence of LOOKAHEAD_WPS waypoints
                for i in range(len(self.base_lane.waypoints)):
                    wp_idx = i
                    if self.car_wp_q >= 0:
                        wp_idx = (self.car_wp_q + i) % len(self.base_lane.waypoints)                    
                    wp = self.base_lane.waypoints[wp_idx]
                    wp_x = wp.pose.pose.position.x
                    wp_y = wp.pose.pose.position.y
                    veh_to_wp_dist = math.sqrt((wp_x - veh_x)**2 + (wp_y - veh_y)**2)
                    if veh_to_wp_dist < min_dist:
                        theta = math.atan2(wp_y - veh_y, wp_x - veh_x)
                        # make sure the waypoint is in front of the car
                        if abs(angles.shortest_angular_distance(theta, veh_yaw)) < math.pi/4.0:
                            state = 1  # found correctly oriented waypoint
                            min_dist = veh_to_wp_dist
                            wp_start = wp_idx
                            # we should be able to stop iterating because the next waypoint should be in front
                            #   of the previous waypoint...for now, starting where we left off should at least
                            #   eliminate a lot of atan2 calls
                    if veh_to_wp_dist > dist_q and state == 1:
                        state = 2  # wp getting farther away from car
                        break;
                    dist_q = veh_to_wp_dist

            if wp_start >= 0:
                self.car_wp_q = wp_start
                braking_range = 100.0  # m to stop the car
                vstep = 0.0
                if self.traffic_wp >= 0:
                    dist_traffic = min(braking_range, self.distance(self.base_lane.waypoints, wp_start, self.traffic_wp))
                    if dist_traffic > 0:
                        vstep = self.target_velocity / dist_traffic  # (m/s)/m deceleration

                # check validity of the traffic_wp
                new_traffic_wp = self.traffic_wp >= 0 and self.traffic_wp != self.last_traffic_wp_processed
                traffic_wp_valid = self.traffic_wp >= wp_start and new_traffic_wp
                if traffic_wp_valid == False and new_traffic_wp:
                    # wraparound
                    if (self.traffic_wp - wp_start) % len(self.base_lane.waypoints) < 100:
                        traffic_wp_valid = True
                
                        
                # RACE condition: flag may be cleared before we see it or may be cleared if thread switches
                #   mid-loop
                process_traffic = (self.traffic_wp < 0) or traffic_wp_valid
                rospy.loginfo("Car wp: %d traffic_wp: %d last_processed: %d process %s",
                              wp_start, self.traffic_wp, self.last_traffic_wp_processed, process_traffic)
                for wp_idx in range(LOOKAHEAD_WPS):
                    idx = (wp_start + wp_idx) % len(self.base_lane.waypoints)
                    # Only set the deacceleration once
                    if process_traffic:
                        velocity = self.target_velocity

                        if traffic_wp_valid and idx <= self.traffic_wp:
                            dist_traffic = self.distance(self.base_lane.waypoints, idx, self.traffic_wp)
                            if dist_traffic <= braking_range:
                                velocity = min(self.target_velocity, vstep * dist_traffic)

                        self.set_waypoint_velocity(self.base_lane.waypoints, idx, velocity)
                        # let last processed go to -1 in case we detect red - green - red on the same signal
                        # since -1 overwrites the velocity, we have to repeat the deceleration
                        self.last_traffic_wp_processed = self.traffic_wp

                    pub_waypoints.append(self.base_lane.waypoints[idx])

                self.wp_start_q = wp_start

                l = Lane()
                l.header.seq = self.seqnum
                self.seqnum = self.seqnum + 1
                l.header.stamp = rospy.get_rostime()
                l.waypoints = pub_waypoints
                self.final_waypoints_pub.publish(l)
                            
#            rospy.loginfo("wp_start %d min_dist %f wp.x %f pose.x %f veh_yaw %f",
 #                         wp_start, min_dist, self.base_lane.waypoints[wp_start].pose.pose.position.x,
  #                        msg.pose.position.x, veh_yaw)

    def waypoints_cb(self, waypoints):
        # Callback for /waypoints message.
        self.base_lane = waypoints
        for idx in range(len(self.base_lane.waypoints)):
            self.set_waypoint_velocity(self.base_lane.waypoints, idx, self.target_velocity)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message.
        self.traffic_wp = msg.data

    def obstacle_cb(self, msg):
        # Callback for /obstacle_waypoint message. We will implement it later
        self.obstacle_wp = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        wp_cnt = (wp2 - wp1 + 1) % len(self.base_lane.waypoints)
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        wp_q = wp1
        for i in range(wp_cnt):
            idx = (wp1 + i) % len(self.base_lane.waypoints)
            dist += dl(waypoints[wp_q].pose.pose.position, waypoints[idx].pose.pose.position)
            wp_q = idx
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
