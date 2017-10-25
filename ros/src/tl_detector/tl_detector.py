#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np
import os

STATE_COUNT_THRESHOLD = 3
COLLECT_TRAINING_DATA = False
EXTRACT_SITE_IMAGES = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.prev_pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.stop_lines = {}

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.is_site = rospy.get_param("/launch") == "site"

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.light_classifier = TLClassifier()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def pose_cb(self, msg):
        self.prev_pose = self.pose
        self.pose = msg

    def car_moved(self):
        if self.prev_pose is None and self.pose is not None:
            return True
        if self.prev_pose.pose.position.x != self.pose.pose.position.x:
            return True
        if self.prev_pose.pose.position.y != self.pose.pose.position.y:
            return True
        return False

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        # List of positions that correspond to the line to stop in front of for a given intersection
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        stop_line_positions = self.config['stop_line_positions']

        self.stop_lines = {}
        for stop_line in stop_line_positions:
            stop_line_pose = Pose()
            stop_line_pose.position.x = stop_line[0]
            stop_line_pose.position.y = stop_line[1]
            line_position = self.get_closest_waypoint(stop_line_pose)
            self.stop_lines[line_position] = stop_line_pose

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        if EXTRACT_SITE_IMAGES:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            filename = os.path.abspath("light_classification/site_training_data/4-{}.jpg".format(rospy.Time.now()))
            cv2.imwrite(filename, cv_image)

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def distance_to_waypoint(self, pose1, pose2):
        veh_x = pose1.x
        veh_y = pose1.y
        wp_x = pose2.x
        wp_y = pose2.y
        dx = veh_x - wp_x
        dy = veh_y - wp_y
        return math.sqrt(dx * dx + dy * dy)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        nearest = None
        min_dist = float("inf")
        for index, waypoint in enumerate(self.waypoints.waypoints):
            dist = self.distance_to_waypoint(pose.position, waypoint.pose.pose.position)
            if dist < min_dist:
                min_dist = dist
                nearest = index

        return nearest


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
        cx = image_width / 2
        cy = image_height / 2

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/world", "/base_link", now, rospy.Duration(1.0))
            tl_point = PointStamped()
            tl_point.header.frame_id = "/world"
            tl_point.header.stamp = now
            tl_point.point.x = point_in_world.x
            tl_point.point.y = point_in_world.y
            tl_point.point.z = point_in_world.z

            tl_point = self.listener.transformPoint("/base_link", tl_point)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException) as e:
            rospy.logerr("Failed to find camera to map transform: {}".format(e))
            return None

        objectPoints = np.array([tl_point.point.y/tl_point.point.x, tl_point.point.z/tl_point.point.x, 1.0])

        ################################################################################
        # Manually tune focal length and camera coordinate for simulator
        #   for more details about this issue, please reference
        #   https://discussions.udacity.com/t/focal-length-wrong/358568/22
        if fx < 10:
            fx = -2580
            fy = -2730
            cx = 360;
            cy = 700;
            objectPoints[2]
        ################################################################################

        # TODO This can be a class member
        cameraMatrix = np.array([[fx, 0,  0],
                                 [0,  fy, 0],
                                 [0,  0,  1]])

        imagePoints = cameraMatrix.dot(objectPoints)
        x = int(round(imagePoints[0]) + cx)
        y = int(round(imagePoints[1]) + cy)
        if 10 > x > image_width-10 or 10 > y > image_height-10:
            return None
        # rospy.loginfo("objectpoints: x: {}, y: {}, z: {}".format(tl_point.point.x, tl_point.point.y, tl_point.point.z))
        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if not light:
            return TrafficLight.UNKNOWN

        if not self.has_image:
            self.prev_light_loc = None
            return False

        crop_size = self.light_classifier.crop_size
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        if self.is_site:
            # remove hood and upper unused sections
            cropped = cv_image[280:750, :, :]
            cropped = cv2.resize(cropped, (crop_size, crop_size))
        else:
            xy = self.project_to_image_plane(light.pose.pose.position)

            if xy is None:
                cropped = cv2.resize(cv_image, (crop_size, crop_size))
            else:
                x, y = xy
                rospy.loginfo("light image loc: {}, {}".format(x, y))
                image_width = self.config['camera_info']['image_width']
                image_height = self.config['camera_info']['image_height']

                half_crop = int(crop_size/2)
                if x > image_width - half_crop:
                    x_min = image_width - crop_size
                    x_max = image_width
                elif x < half_crop:
                    x_min = 0
                    x_max = crop_size
                else:
                    x_min = max(0, x-half_crop)
                    x_max = min(x_min+crop_size, image_width)

                if y > image_height - half_crop:
                    y_min = image_height - crop_size
                    y_max = image_height
                elif y < half_crop:
                    y_min = 0
                    y_max = crop_size
                else:
                    y_min = max(0, y-half_crop)
                    y_max = min(y_min+crop_size, image_height)

                cropped = cv_image[y_min:y_max, x_min:x_max]

        pred = self.light_classifier.get_classification(cropped)

        rospy.loginfo("Truth: {}, Pred: {}".format(light.state, pred))

        if light.state != TrafficLight.UNKNOWN and self.car_moved() and COLLECT_TRAINING_DATA:
            # Ground truth is known, use it to create training data
            filename = os.path.abspath("light_classification/training_data/{}-{}.jpg".format(light.state, rospy.Time.now()))
            rospy.loginfo("Car moved, saving new image: {}".format(filename))
            cv2.imwrite(filename, cropped)

        return pred

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.waypoints or not self.pose:
            return -1, TrafficLight.UNKNOWN

        light = None

         # car_fwd indicates whether car's moving the same direction as the waypoint index increase
        car_fwd = False
        closest_wp_index = None
        waypoints = self.waypoints.waypoints
        wp_length = len(waypoints)
        if self.pose:
            car_orientation = self.pose.pose.orientation
            quaternion = (car_orientation.x, car_orientation.y,
                          car_orientation.z, car_orientation.w)
            _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)

            car_pose = self.pose.pose
            closest_wp_index = self.get_closest_waypoint(car_pose)
            wp1 = waypoints[closest_wp_index]
            wp2 = waypoints[(closest_wp_index + 1) % wp_length]
            fwd_angle = math.atan2(wp2.pose.pose.position.y - wp1.pose.pose.position.y,
                                   wp2.pose.pose.position.x - wp1.pose.pose.position.x)

            if -math.pi / 2.0 < yaw - fwd_angle < math.pi / 2:
                car_fwd = True

        light, light_wp = None, None

        for i in range(wp_length):
            inc = 1 if car_fwd else -1
            index = (closest_wp_index + i * inc) % wp_length
            if index in self.stop_lines:
                light_wp = index
                stop_line_pose = self.stop_lines[index]
                min_dist = float("inf")
                # Find associated light
                for light_candidate in self.lights:
                    dist = self.distance_to_waypoint(light_candidate.pose.pose.position, stop_line_pose.position)
                    if dist < min_dist:
                        light = light_candidate
                        min_dist = dist
                break

        if light:
            state = TrafficLight.UNKNOWN
            if abs(light_wp - closest_wp_index) < 50:
                state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
