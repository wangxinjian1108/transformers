# /usr/bin/python3
import logging
import os, sys
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union, Type, Dict, overload
from datetime import datetime, timezone, timedelta
import concurrent.futures
import yaml
from urllib.request import urlopen
from urllib.parse import urlparse, unquote
import requests
from collections import OrderedDict
import cv2
from cv_bridge import CvBridge
import os
from os import path as osp
from perception.lane_detection_pb2 import LaneDetection
from perception.obstacle_detection_pb2 import ObstacleDetection, PerceptionObstacle
from monitor.status_report_msg_pb2 import EnvironmentState, StatusReport
from localization.localization_pb2 import LocalizationEstimation
from sensor_calibration.sensor_calibration_pb2 import CalibrationMessage
from pluspy.bag_utils import is_ros_bag, extract_bag_components, bag_open_close
import fastbag
from functools import partial
import json
from enum import IntEnum
import numpy as np
import time
import threading


# Configure the logger
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='[%(filename)s:%(lineno)d:%(levelname)s]:%(message)s -',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# logger = logging.getLogger(__name__)
logger = None

image_topics = [
    '/front_left_camera/image_color/compressed',
    '/front_center_camera/image_color/compressed',
    '/front_right_camera/image_color/compressed',
    '/side_left_camera/image_color/compressed',
    '/side_right_camera/image_color/compressed',
    '/rear_left_camera/image_color/compressed',
    '/rear_right_camera/image_color/compressed'
]

localization_topic = "/localization/state"
localization_gnss_topic = "/localization/gnss"
lane_detection_topic = "/perception/lane_path"
obstacle_detection_topic = "/perception/obstacles"
localization_status_topic = "/localization/status_report"
old_perception_status_topic = "/obstacle/status_report"
perception_status_topic = "/perception/status_report"
    
def in_roi(x, y, args: argparse.Namespace):
    return args.min_x < x < args.max_x and args.min_y < y < args.max_y

def in_side_region(x, y, args: argparse.Namespace):
    return args.min_side_x < x < args.max_side_x and args.min_side_y < y < args.max_side_y

def is_side_potential_fp(x, y, tracking_time, occluded=False):
    return -10 < x < 15 and abs(y) > 2 and tracking_time < 0.5 and not occluded

def parse_protobuf_message(data: str, message_type):
    message = message_type()
    message.ParseFromString(data[1].data)
    return message

def parse_env_states_message(data: str):
    message = StatusReport()
    message.ParseFromString(data[1].data)
    return message.environment_states


def datastr_to_timestamp(datastr):
    dt = datetime.strptime(datastr, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone(timedelta(hours=8)))
    timestamp = dt.timestamp()
    return timestamp

def timestamp_to_datastr(timestamp):
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    dt_east8 = dt.astimezone(timezone(timedelta(hours=8)))
    return dt_east8.isoformat()

def timestamp_to_formatstr(timestamp, format_str):
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    dt_east8 = dt.astimezone(timezone(timedelta(hours=8)))
    return dt_east8.strftime(format_str)

def parseAndWriteCalibData(calib_data, output_dir, calib_infos):
    calib_output_dir = output_dir + "/calib"
    os.system("mkdir -p {}".format(calib_output_dir))
    tmp_file = calib_output_dir + "/tmp.yml"
    f = open(tmp_file, "w")
    f.write(calib_data)
    f.close()
    yaml_file = cv2.FileStorage(tmp_file, cv2.FILE_STORAGE_READ)
    vehicle_name = yaml_file.getNode("car").string()
    date = yaml_file.getNode("date").string()
    sensor_name = yaml_file.getNode("sensor_name").string()
    if sensor_name not in calib_infos:
        calib_file_name = calib_output_dir + "/" + vehicle_name + "_" + date + "_" + sensor_name + ".yml"
        os.rename(tmp_file, calib_file_name)
        # print("write calib file: ", vehicle_name + "_" + date + "_" + sensor_name + ".yml", " to ", calib_output_dir)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    return vehicle_name, date, sensor_name


def get_env_status_value(env_states, key):
    for env_state in env_states:
        if env_state.state_type == key:
            return env_state.value
    return None


def drop_nan_and_take_most_frequent(data, default_val):
    data = [d for d in data if d is not None]
    if len(data) == 0:
        return default_val
    res = max(set(data), key=data.count)
    if isinstance(res, default_val.__class__):
        return res
    return default_val.__class__(res)


BAG_PATH_LOCKS = {}

def set_bag_lock(filepath):
    if filepath not in BAG_PATH_LOCKS:
        BAG_PATH_LOCKS[filepath] = threading.Lock()

def get_bag_lock(filepath):
    if filepath not in BAG_PATH_LOCKS:
        BAG_PATH_LOCKS[filepath] = threading.Lock()
    return BAG_PATH_LOCKS[filepath]


class Indensity(IntEnum):
    NONE = 0
    LIGHT = 1
    HEAVY = 2
    

class EnvState(IntEnum):
    RAIN = 0 # 0-No/1-Light(we could handle)/2-Heavy(case we couldn't handle)
    SNOW = 1 # 0-No/1-Light(we could handle)/2-Heavy(case we couldn't handle)
    MAP_CURVATURE = 400
    MAP_PITCH = 401
    NEAR_TUNNEL = 403 # refer: https://github.pluscn.cn/PlusAI/common_protobuf/blob/d926b8e7f7208257c497a74df4c088e424d269c4/monitor/status_report_msg.proto
    TUNNEL_AHEAD = 506 # the distance to the ahead tunnel, if you are in the tunnel, this value will be always 0.0
    LANE_TYPE = 511 # refer: https://github.pluscn.cn/PlusAI/common_protobuf/blob/879c0956ab32df9dd85866e18e598ec78d744daa/plusmap/lane_geometry.proto#L115 
    COLLISION_RISK = 1501 # int, -1 - no collision risk, 1 - collision risk
    

class LANE_TYPE(IntEnum):
    DEFAULT = 1
    BRIDGE = 2
    CITY_DRIVING = 3
    HIGHWAY = 4
    JUNCTION = 5
    RAMP = 6
    TUNNEL = 7
    SERVICE_DISTRICT = 8
    UNKNOWN = 9
    EMERGENCY_LANE = 10


@dataclass
class ScenarioStatus:
    # time info
    time_stamp: float = 0
    duration: float = 0
    # weather info
    rain_status: Indensity = Indensity.NONE
    snow_status: Indensity = Indensity.NONE
    # obstacle info
    tracking_frames: int = 0
    obs_num: int = 0
    side_obs_num: int = 0
    side_large_obs_num: int = 0
    side_large_potential_fp_num: int = 0
    # ego status
    max_pitch_correction: float = 0 # on degree
    is_pitch_correction_robust: bool = True
    traj_length: float = 0
    ave_speed: float = 0
    max_acceleration: float = 0
    lane_change: bool = False
    # road info
    lane_type: LANE_TYPE = LANE_TYPE.UNKNOWN
    in_tunnel: bool = False # true if at least 3 seconds in tunnel
    max_road_curvature: float = 0
    max_road_pitch: float = 0
    fail_reason: str = ''
    # non-vehicle type: pedestrian, cone, moto, cyclist
    bycycle_frames: int = 0
    pedestrian_frames: int = 0
    cone_frames: int = 0
    moto_frames: int = 0
    
    def non_vehicle_ob_frames(self):
        return self.bycycle_frames + self.pedestrian_frames + self.cone_frames + self.moto_frames
    
    def to_json(self):
        return {
            'time_stamp': timestamp_to_formatstr(self.time_stamp, '%Y%m%dT%H%M%S'),
            'duration': self.duration,
            'rain_status': self.rain_status.name,
            'snow_status': self.snow_status.name,
            'tracking_frames': self.tracking_frames,
            'obs_num': self.obs_num,
            'side_obs_num': self.side_obs_num,
            'side_large_obs_num': self.side_large_obs_num,
            'side_large_potential_fp_num': self.side_large_potential_fp_num,
            'max_pitch_correction': self.max_pitch_correction,
            'is_pitch_correction_robust': self.is_pitch_correction_robust,
            'traj_length': self.traj_length,
            'ave_speed': self.ave_speed,
            'max_acceleration': self.max_acceleration,
            'lane_change': self.lane_change,
            'lane_type': self.lane_type.name,
            'in_tunnel': self.in_tunnel,
            'max_road_curvature': self.max_road_curvature,
            'max_road_pitch': self.max_road_pitch,
            'bycycle_frames': self.bycycle_frames,
            'pedestrian_frames': self.pedestrian_frames,
            'cone_frames': self.cone_frames,
            'moto_frames': self.moto_frames,
            "non_vehicle_frames" : self.non_vehicle_ob_frames()
        }
    
    def __str__(self) -> str:
        return self.to_json().__str__()
    
    def is_suited_for_auto_labeling(self, args: argparse.Namespace):
        if not self.is_pitch_correction_robust and args.use_pitch_correction:
            self.fail_reason = 'pitch correction not robust'
            return False
        hour = datetime.fromtimestamp(self.time_stamp).hour
        if self.ave_speed < args.min_ave_speed or self.ave_speed > args.max_ave_speed:
            self.fail_reason = f'ave speed too small {self.ave_speed:.2f} or too large {self.ave_speed:.2f}'
            return False
        if hour < args.min_hour or hour > args.max_hour:
            self.fail_reason = f'hour not in range {args.min_hour}-{args.max_hour}'
            return False
        # vehicle number check
        if self.side_large_obs_num < args.min_side_large_obs_num and not args.mine_non_vehicle_obstacle:
            self.fail_reason = f'side large obs num too small, {self.side_large_obs_num}'
            return False
        
        if args.mine_non_vehicle_obstacle:
            if self.cone_frames < args.min_cone_frames and self.pedestrian_frames < args.min_pedestrian_frames and \
                self.moto_frames < args.min_moto_frames and self.bycycle_frames < args.min_bycycle_frames:
                self.fail_reason = f'non-vehicle obstacle frames too small'
                return False
        return True


@dataclass
class ClipInfo:
    id: int
    bag_paths: List[str]
    start_times: List[float]
    end_times: List[float]
    scenario_status: ScenarioStatus = ScenarioStatus()
    is_valid_candidate: bool = False
    vehicle_name: str = ''
    status_lock: Optional[threading.Lock] = None
    
    def __str__(self):
        return "{}, bag_paths: {}, mid_time  : {}, duration: {:.1f}".format(self.name(),
            self.bag_paths, self.mid_time(), self.duration())
        
    def __post_init__(self):
        assert len(self.bag_paths) == len(self.start_times) == len(self.end_times), 'bag_paths, start_times, end_times length not match'
        self.status_lock = threading.Lock()
        
    def cross_bag(self):
        return len(self.bag_paths) > 1
    # COMMENT: A clip can be composed of multiple bags, so we need to consider the start time and end time of each bag.
    # It is ensured that there is no overlapping information between two consecutive bags.(By Wenjun)
    
    def start_time(self):
        return self.start_times[0]
    
    def end_time(self):
        return self.end_times[-1]
    
    def mid_time(self):
        return (self.start_time() + self.end_time()) / 2
    
    def duration(self):
        return self.end_time() - self.start_time()
    
    def name(self):
        # mid time as clip name
        id_str = str(self.id).zfill(4)
        ts = timestamp_to_formatstr(self.mid_time(), '%Y%m%dT%H%M%S')
        return f'{self.vehicle_name}_{ts}_{id_str}'
    
    def fuse_clip(self, clip):
        if len(clip.bag_paths) == len(self.bag_paths) and len(self.bag_paths) == 1 and clip.bag_paths[0] == self.bag_paths[0]:
            self.bag_paths
            self.end_times[0] = clip.end_times[0]
        else:
            self.bag_paths.extend(clip.bag_paths)
            self.start_times.extend(clip.start_times)
            self.end_times.extend(clip.end_times)
                        
    def evaluate_clip_status(self, args: argparse.Namespace):
        lane_res, local_res = [], []
        localization_status_report, perception_status_report = [], []
        obstacle_ids, side_ob_ids, side_large_ob_ids, side_large_potential_fp_ids = set(), set(), set(), set()
        
        self.scenario_status.time_stamp = self.mid_time()
        self.scenario_status.duration = self.duration()
        self.scenario_status.tracking_frames = 0
        self.scenario_status.cone_frames = 0
        self.scenario_status.pedestrian_frames = 0
        self.scenario_status.bycycle_frames = 0
        self.scenario_status.moto_frames = 0
        
        # /mnt/vault25/drives/2025-03-03/20250303T045402_pdb-l4e-c0002_2.db
        # 1740951408.588132
        # 1740951414.588132
        
        for bf, st, et in zip(self.bag_paths, self.start_times, self.end_times):
            # bf = "/mnt/vault25/drives/2025-03-03/20250303T045402_pdb-l4e-c0002_2.db"
            # st, et = 1740951408.588132, 1740951414.588132
            # if True:
            with get_bag_lock(bf):  # 加锁
                if not fastbag.Reader.is_readable(bf):
                    logger.error(f'bag {bf} is not readable, path exists: {os.path.exists(bf)}')
                    continue
                with fastbag.Reader(bf) as reader:
                    # 1. read obstacle info
                    for aa, msg, cc in reader.read_messages(obstacle_detection_topic, start_time=st, end_time=et):
                        obs_pb = ObstacleDetection()
                        obs_pb.ParseFromString(msg.data)
                        if len(obs_pb.obstacle) == 0:
                            continue
                        # print(f"fuck mingyao {self.scenario_status.tracking_frames}, timestamp {aa}, {cc}")
                        
                        track_ids = [ob.id for ob in obs_pb.obstacle]
                        tracking_times = [ob.tracking_time for ob in obs_pb.obstacle]
                        xs = [ob.motion.xrel for ob in obs_pb.obstacle]
                        ys = [ob.motion.yrel for ob in obs_pb.obstacle]
                        status_valids = [ob.track_maturity == 5 for ob in obs_pb.obstacle]
                        in_rois = [in_roi(x, y, args) for x, y in zip(xs, ys)]
                        in_sides = [in_side_region(x, y, args) for x, y in zip(xs, ys)]
                        is_vehicles = [ob.length > 3 and ob.width > 1 and ob.height > 1 for ob in obs_pb.obstacle]
                        is_large_obs = [ob.length > 10 and ob.width > 1 and ob.height > 1 for ob in obs_pb.obstacle]
                        is_cones = [int(ob.type == PerceptionObstacle.CONE) for ob in obs_pb.obstacle]
                        is_pedestrians = [int(ob.type == PerceptionObstacle.PEDESTRIAN) for ob in obs_pb.obstacle]
                        is_bycycles = [int(ob.type == PerceptionObstacle.BICYCLE) for ob in obs_pb.obstacle]
                        is_motos = [int(ob.type == PerceptionObstacle.MOTO) for ob in obs_pb.obstacle]
                        # types = [ob.type for ob in obs_pb.obstacle]
                        
                        valid_obs = [int(in_roi and is_vehicle and status_valid) for in_roi, is_vehicle, status_valid in zip(in_rois, is_vehicles, status_valids)]
                        side_obs = [int(in_side and is_vehicle and status_valid) for in_side, is_vehicle, status_valid in zip(in_sides, is_vehicles, status_valids)]
                        side_large_obs = [int(in_side and large and status_valid) for in_side, large, status_valid in zip(in_sides, is_large_obs, status_valids)]
                        obstacle_ids |= set([track_id for track_id, valid in zip(track_ids, valid_obs) if valid])
                        side_ob_ids |= set([track_id for track_id, valid in zip(track_ids, side_obs) if valid])
                        side_large_ob_ids |= set([track_id for track_id, valid in zip(track_ids, side_large_obs) if valid])
                        side_large_potential_fp_ids |= set([track_id for track_id, x, y, tracking_time, is_large_ob in zip(track_ids, xs, ys, tracking_times, is_large_obs)
                                                            if is_side_potential_fp(x, y, tracking_time) and is_large_ob])
                        
                        with self.status_lock:
                            self.scenario_status.tracking_frames += 1
                            self.scenario_status.bycycle_frames += int(sum(is_bycycles) > 0)
                            self.scenario_status.pedestrian_frames += int(sum(is_pedestrians) > 0)
                            self.scenario_status.cone_frames += int(sum(is_cones) > 2)
                            self.scenario_status.moto_frames += int(sum(is_motos) > 0)
                        
                    # 2. read localization info
                    lane_res.extend(list(reader.read_messages(lane_detection_topic, start_time=st, end_time=et)))
                    local_res.extend(list(reader.read_messages(localization_topic, start_time=st, end_time=et)))
                    localization_status_report.extend(list(reader.read_messages(localization_status_topic, start_time=st, end_time=et)))
                    tmp_perception_status_report = list(reader.read_messages(perception_status_topic, start_time=st, end_time=et))
                    if len(tmp_perception_status_report) == 0:
                        tmp_perception_status_report = list(reader.read_messages(old_perception_status_topic, start_time=st, end_time=et))
                    perception_status_report.extend(tmp_perception_status_report)
        
        if self.scenario_status.tracking_frames > self.duration() * 10 * 1.25:
            logger.error(f'Found multi-thread process tracking frames count error, tracking frames: {self.scenario_status.tracking_frames}, {self.__str__()}')
                
        self.scenario_status.obs_num = len(obstacle_ids)
        self.scenario_status.side_obs_num = len(side_ob_ids)
        self.scenario_status.side_large_obs_num = len(side_large_ob_ids)
        self.scenario_status.side_large_potential_fp_num = len(side_large_potential_fp_ids)
        if self.scenario_status.obs_num == 0:
            return
        
        # lane detection
        parse_lane_detection = partial(parse_protobuf_message, message_type=LaneDetection)
        lane_detections = list(map(parse_lane_detection, lane_res))
        pitchs = [ld.runtime_pose_angle_correction.pitch for ld in lane_detections]
        pitch_modes = [ld.pitch_estimator_mode for ld in lane_detections]
        self.scenario_status.max_pitch_correction = max(pitchs) * 180 / np.pi
        self.scenario_status.is_pitch_correction_robust = all([mode == 1 for mode in pitch_modes])
        
        # localization info
        parse_localization_estimation = partial(parse_protobuf_message, message_type=LocalizationEstimation)
        localization_estimations = list(map(parse_localization_estimation, local_res))
        accs = [[le.linear_acceleration.x, le.linear_acceleration.y, le.linear_acceleration.z] for le in localization_estimations]
        accs = [np.linalg.norm(acc) for acc in accs]
        self.scenario_status.max_acceleration = max(accs)
        x_diff = localization_estimations[-1].odometry_position.x - localization_estimations[0].odometry_position.x
        y_diff = localization_estimations[-1].odometry_position.y - localization_estimations[0].odometry_position.y
        self.scenario_status.traj_length = ((x_diff ** 2 + y_diff ** 2) ** 0.5)
        self.scenario_status.ave_speed = self.scenario_status.traj_length / self.duration()
        
        # status report
        perception_env_states = list(map(parse_env_states_message, perception_status_report))
        localization_env_states = list(map(parse_env_states_message, localization_status_report))
        rain_status = [get_env_status_value(env_state, EnvState.RAIN) for env_state in perception_env_states]
        snow_status = [get_env_status_value(env_state, EnvState.SNOW) for env_state in perception_env_states]
        curvature = [get_env_status_value(env_state, EnvState.MAP_CURVATURE) for env_state in localization_env_states]
        pitch = [get_env_status_value(env_state, EnvState.MAP_PITCH) for env_state in localization_env_states]
        lane_types = [get_env_status_value(env_state, EnvState.LANE_TYPE) for env_state in localization_env_states]
        tunnel_aheads = [get_env_status_value(env_state, EnvState.TUNNEL_AHEAD) for env_state in localization_env_states]
        self.scenario_status.rain_status = drop_nan_and_take_most_frequent(rain_status, Indensity.NONE)
        self.scenario_status.snow_status = drop_nan_and_take_most_frequent(snow_status, Indensity.NONE)
        self.scenario_status.max_road_curvature = max([c for c in curvature if c is not None] + [0])
        self.scenario_status.max_road_pitch = max([p for p in pitch if p is not None] + [0])
        self.scenario_status.lane_type = drop_nan_and_take_most_frequent(lane_types, LANE_TYPE.UNKNOWN)
        self.scenario_status.in_tunnel = tunnel_aheads.count(0) > 30
       
        self.is_valid_candidate = self.scenario_status.is_suited_for_auto_labeling(args)
        fail_reason = f', due to {self.scenario_status.fail_reason}' if not self.is_valid_candidate else ''
        logger.info(f'{self.name()} valid: {self.is_valid_candidate}{fail_reason}')
        
    def to_json(self):
        js = self.scenario_status.to_json()
        js['id'] = self.id
        js["bag_paths"] = ",".join(self.bag_paths)
        js["start_times"] = ",".join([str(st) for st in self.start_times])
        js["end_times"] = ",".join([str(et) for et in self.end_times])
        js["is_valid_candidate"] = self.is_valid_candidate
        return js

    def download_info(self, save_dir):
        logger.debug(f'start to parse info for clip: {self.name()} ...')
        # 1. read images
        lane_res, local_res, local_gnss_res = [], [], []
        for bf, st, et in zip(self.bag_paths, self.start_times, self.end_times):
            with fastbag.Reader(bf) as reader:
                # lane detection && localization info
                lane_res.extend(list(reader.read_messages(lane_detection_topic, start_time=st, end_time=et)))
                local_res.extend(list(reader.read_messages(localization_topic, start_time=st, end_time=et)))
                local_gnss_res.extend(list(reader.read_messages(localization_gnss_topic, start_time=st, end_time=et)))
                # 1. write images
                for topic in image_topics:
                    nickname = topic.split("/")[1]
                    topic_dir = osp.join(save_dir, "raw_images", nickname)
                    os.makedirs(topic_dir, exist_ok=True)
                    for _, img, _ in reader.read_messages(topic, start_time=st, end_time=et):
                        cv_img = CvBridge().compressed_imgmsg_to_cv2(img)
                        if cv_img.size == 0 or cv_img is None:
                            continue
                        timestamp = img.header.stamp.to_sec()
                        imgp = osp.join(topic_dir, "{:.6f}.png".format(timestamp))
                        cv2.imwrite(imgp, cv_img)
                    logger.debug(f'write images to {topic_dir}')
        # 2. write lane detection
        # element: u'/perception/lane_path', string message ,genpy.Time[1680079372285041809]
        # convert string to LaneDetection
        parse_lane_detection = partial(parse_protobuf_message, message_type=LaneDetection)
        lane_detections = list(map(parse_lane_detection, lane_res))
        pitchs = [ld.runtime_pose_angle_correction.pitch for ld in lane_detections]
        pitch_modes = [ld.pitch_estimator_mode for ld in lane_detections]
        # timestamps = [i[2].secs + i[2].nsecs * 1e-9 for i in lane_res]
        timestamps = [ld.header.timestamp_msec / 1000 for ld in lane_detections]
        pitch_results = list(zip(timestamps, pitchs, pitch_modes))
        pitch_results = sorted(pitch_results) # sort as time stamp
        with open(osp.join(save_dir, "pitch_results.txt"), "w") as f:
            for ts, pitch, mode in pitch_results:
                f.write("{:.6f} {:.9f} {}\n".format(ts, pitch, mode))
        # 3. write localization info
        parse_localization_estimation = partial(parse_protobuf_message, message_type=LocalizationEstimation)
        localization_estimations = list(map(parse_localization_estimation, local_res))
        timestamps = [le.header.timestamp_msec / 1000 for le in localization_estimations]
        positions = [[le.odometry_position.x, le.odometry_position.y, le.odometry_position.z] for le in localization_estimations]
        quaterions = [[le.odometry_orientation.qx, le.odometry_orientation.qy, le.odometry_orientation.qz, le.odometry_orientation.qw] \
                        for le in localization_estimations]
        yaws = [[le.odometry_euler_angles.yaw] for le in localization_estimations]
        speeds = [[le.odometry_linear_velocity.x, le.odometry_linear_velocity.y, le.odometry_linear_velocity.z] for le in localization_estimations]
        fusion_localization_results = list(zip(timestamps, positions, yaws, quaterions, speeds)) # angulars, accels)
        fusion_localization_results = sorted(fusion_localization_results)
        with open(osp.join(save_dir, "localization_results.txt"), "w") as f:
            for ts, pos, yaw, quat, speed in fusion_localization_results:
                f.write("{:.6f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(ts, pos[0], pos[1], pos[2], yaw[0],
                                quat[0], quat[1], quat[2], quat[3], speed[0], speed[1], speed[2]))
        # 4. read localization gnss info
        local_gnss_res = list(reader.read_messages(localization_gnss_topic))
        parse_localization_estimation = partial(parse_protobuf_message, message_type=LocalizationEstimation)
        localization_estimations = list(map(parse_localization_estimation, local_gnss_res))
        timestamps = [le.header.timestamp_msec / 1000 for le in localization_estimations]
        positions = [[le.position.x, le.position.y, le.position.z] for le in localization_estimations]
        quaterions = [[le.orientation.qx, le.orientation.qy, le.orientation.qz, le.orientation.qw] \
                        for le in localization_estimations]
        yaws = [[le.euler_angles.yaw] for le in localization_estimations]
        speeds = [[le.linear_velocity.x, le.linear_velocity.y, le.linear_velocity.z] for le in localization_estimations]
        fusion_localization_results = list(zip(timestamps, positions, yaws, quaterions, speeds)) # angulars, accels)
        fusion_localization_results = sorted(fusion_localization_results)
        with open(osp.join(save_dir, "localization_gnss_results.txt"), "w") as f:
            for ts, pos, yaw, quat, speed in fusion_localization_results:
                f.write("{:.6f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(ts, pos[0], pos[1], pos[2], yaw[0],
                                quat[0], quat[1], quat[2], quat[3], speed[0], speed[1], speed[2]))
        logger.info(f'finished parsing info for clip: {self.name()}')
        
        
    def process(self, valid_dir, invalid_dir, args: argparse.Namespace):
        valid_clip_dir = osp.join(valid_dir, self.name())
        invalid_clip_dir = osp.join(invalid_dir, self.name())
        if os.path.exists(valid_clip_dir) and os.path.exists(osp.join(valid_clip_dir, "clip_info.json")) and args.check_indicator:
            logger.warning(f'clip {self.name()} has been processed')
            return
        if os.path.exists(invalid_clip_dir) and os.path.exists(osp.join(invalid_clip_dir, "clip_info.json")) and args.check_indicator:
            logger.warning(f'clip {self.name()} has been processed')
            return
        
        self.evaluate_clip_status(args)
        save_dir = valid_clip_dir if self.is_valid_candidate else invalid_clip_dir
        os.makedirs(save_dir, exist_ok=True)
        if args.debug_mode:
            with open(osp.join(save_dir, "clip_info.json"), "w") as f:
                json.dump(self.to_json(), f, indent=4)
            return
        
        if self.is_valid_candidate:
            # os.makedirs(osp.join(save_dir, "calib"), exist_ok=True)
            # os.system(f'cp {output_dir}/calib/* {save_dir}/calib')
            self.download_info(save_dir)
        with open(osp.join(save_dir, "clip_info.json"), "w") as f:
            json.dump(self.to_json(), f, indent=4)


@dataclass
class BagInfo:
    path: str
    start_time: float
    end_time: float
    distance: float
    speed: float
    duration: float = 0
    start_date: str = ''
    end_date: str = ''
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time
        self.start_date = timestamp_to_formatstr(self.start_time, '%Y-%m-%d %H:%M:%S')
        self.end_date = timestamp_to_formatstr(self.end_time, '%Y-%m-%d %H:%M:%S')
        set_bag_lock(self.path)

    def __str__(self):
        return "BagInfo: {}, duration: {:.0f}, distance: {:.0f}, speed: {:.1f}".format(
            self.path.split("/")[-1], self.duration, self.distance, self.speed)
        
    def fragment_one_clip(self, start_time, duration, id) -> ClipInfo:
        return ClipInfo(id=id, bag_paths=[self.path], start_times=[start_time], end_times=[min(start_time + duration, self.end_time)])
    
    def fragment_clips(self, start_time, duration, first_id, veh_name) -> List[ClipInfo]:
        clips = []
        cur_st, cur_et = start_time, start_time + duration
        while cur_et < self.end_time:
            clips.append(ClipInfo(id=len(clips) + first_id, bag_paths=[self.path], start_times=[cur_st], end_times=[cur_et], vehicle_name=veh_name))
            cur_st, cur_et = cur_et, cur_et + duration
        if cur_st < self.end_time:
            clips.append(ClipInfo(id=len(clips) + first_id, bag_paths=[self.path], start_times=[cur_st], end_times=[self.end_time], vehicle_name=veh_name))
        return clips
        

@dataclass
class TripInfo:
    bags: List[BagInfo]
    start_time: float = 0
    end_time: float = 0
    duration: float = 0
    distance: float = 0
    speed: float = 0
    start_date: str = ''
    end_date: str = ''
    vehicle_name: str = ''
    clips: List[ClipInfo] = field(default_factory=list)
    
    def __post_init__(self):
        self.start_time = self.bags[0].start_time
        self.end_time = self.bags[-1].end_time
        self.duration = self.end_time - self.start_time
        self.distance = sum([clip.distance for clip in self.bags])
        self.speed = self.distance / self.duration
        self.start_date = timestamp_to_formatstr(self.start_time, '%Y-%m-%d %H:%M:%S')
        self.end_date = timestamp_to_formatstr(self.end_time, '%Y-%m-%d %H:%M:%S')
        
    def __str__(self):
        return "TripInfo: from {} to {}, {} bags, duration: {:.1f}h, distance: {:.1f} km, speed: {:.1f}".format(
            self.start_date, self.end_date, len(self.bags), self.duration / 3600., self.distance / 1000, self.speed)
    
    def bags_info(self):
        return "\n".join([str(bag) for bag in self.bags])
        
    def name(self):
        st = timestamp_to_formatstr(self.start_time, '%Y%m%dT%H%M%S')
        et = timestamp_to_formatstr(self.end_time, '%Y%m%dT%H%M%S')
        return f'{self.vehicle_name}_{st}_{et}'
        
    def fragment_clips(self, duration) -> List[ClipInfo]:
        results = []
        ts = -1
        for i in range(len(self.bags)):
            bag = self.bags[i]
            clips = bag.fragment_clips(start_time=bag.start_time if ts < 0 else ts, duration=duration, first_id=len(results), veh_name=self.vehicle_name)
            assert len(clips) > 0, 'no clips for bag: {}'.format(bag)
            results.extend(clips)
            if results[-1].duration() < duration and i + 1 < len(self.bags):
                next_duration = duration - results[-1].duration()
                ts = self.bags[i + 1].start_time + next_duration
                results[-1].fuse_clip(self.bags[i + 1].fragment_one_clip(start_time=self.bags[i + 1].start_time, duration=next_duration, id=len(results)))
        results[-2].fuse_clip(results[-1])
        self.clips = results[:-1]
        logger.warning(f'found total {len(self.clips)} clips for trip: {self.name()}')
        
    def download_calib(self, output_dir):
        src_bag = None
        for bag in self.bags:
            if fastbag.Reader.is_readable(bag.path):
                src_bag = fastbag.Reader(bag.path)
                break
        assert src_bag is not None, 'no readable bag found for trip: {}'.format(self)
        calib_infos = {}
        src_bag.open()
        bag_iter = src_bag.iter_messages(["/perception/calibrations"])
        _, msg, _ = next(bag_iter)
        calib_msg = CalibrationMessage()
        calib_msg.ParseFromString(msg.data)
        
        for calib_data in calib_msg.data:
            vehicle_name, date, sensor_name = parseAndWriteCalibData(calib_data.calib_data, output_dir, calib_infos)
            calib_infos[sensor_name] = vehicle_name + "_" + date
        # src_bag.close()
        
    def process_clip(self, clip_id, output_dir, args: argparse.Namespace):
        assert clip_id < len(self.clips), 'clip_id out of range: {}'.format(clip_id)
        self.clips[clip_id].process(f'{output_dir}/valid_clips', f'{output_dir}/invalid_clips', args)
        return self.clips[clip_id].is_valid_candidate


class BatchTripTaskManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.trips = self.cluster_bags_to_trip(self.query_infos(args), args.vehicle_name)
        self.trips = [self.trips[0]]
        self.indexs = [] # trip index -> clip index
        logger.error(f'found total {len(self.trips)} trips for vehicle: {args.vehicle_name} from {args.start_date} to {args.end_date}')
    
    @staticmethod
    def query_infos(args: argparse.Namespace):
        query_string = 'https://bagdb.pluscn.cn/api/v1/bags?order=-time&page=1&page_size=10000&start_time=2023-11-29%2000%3A15&end_time=2023-11-29%2008%3A31&vehicleIs=pdb-l4e-c0002&bag_source=offline&drive_info_node=%2A'
        decoded_url = unquote(query_string)
        logger.debug('decoded_url  : {}'.format(decoded_url))
        
        start_timestamp = datastr_to_timestamp(args.start_date + ' ' + args.start_time)
        end_timestamp = datastr_to_timestamp(args.end_date + ' ' + args.end_time)

        utc_delta_seconds = 8 * 60 * 60
        utc_start_timestamp = start_timestamp - utc_delta_seconds
        utc_end_timestamp = end_timestamp - utc_delta_seconds
        
        utc_start_timestr = timestamp_to_formatstr(utc_start_timestamp, '%Y-%m-%d %H:%M:%S')
        utc_end_timestr = timestamp_to_formatstr(utc_end_timestamp, '%Y-%m-%d %H:%M:%S')
        query_string = 'https://bagdb.pluscn.cn/api/v1/bags?order=-time&page=1&page_size=10000&start_time={}&end_time={}&vehicleIs={}&bag_source=offline&drive_info_node=*'.format(
            utc_start_timestr, utc_end_timestr, args.vehicle_name
        )
        logger.debug('query_string : {}'.format(query_string))

        response = requests.get(query_string, allow_redirects=True)
        baginfos_content = response.content.decode("utf-8")
        baginfos = yaml.safe_load(baginfos_content)
        baginfos = sorted(baginfos, key=lambda x: x['start_time'])
        
        filtered_baginfos = []
        # <<<< comment: single bag info be like >>>>
        
        # baginfos[0].keys()
        # dict_keys(['id', 'bag_name', 'rosbag_status', 'bag_path', 'fastbag_path', 'fastbag_version', 'vehicle', 'bag_source', 
        #            'drive_info', 'mode', 'start_time', 'end_time', 'start_location', 'end_location', 'distance', 'plusview_uri', 
        #            'camera_view_uri', 'plusview_error', 'has_online_events', 'events_count', 'events_processing_km', 'events_processing_count'])
        
        # {'id': 2628892, 'bag_name': '20240601T062436_pdb-l4e-c0010_0.bag', 'rosbag_status': 'normal', 
        # 'bag_path': 'https://bagdb.pluscn.cn:28443/raw/mnt/vault20/drives/2024-06-01/20240601T062436_pdb-l4e-c0010_0.bag?_x_user=bagdb@plus.ai', 
        # 'fastbag_path': 'https://bagdb.pluscn.cn:28443/raw/mnt/vault46/drives/2024-06-01/20240601T062436_pdb-l4e-c0010_0.db?_x_user=bagdb@plus.ai', 
        # 'fastbag_version': '0.0.0', 'vehicle': 'pdb-l4e-c0010', 'bag_source': 'offline', 'drive_info': {'build_tag': '2.2.20578', 'build_branch': 
        # 'acc-test-20240527', 'build_commit': 'abf636b', 'build_number': '20578'}, 'mode': '63', 'start_time': 1717194279.0615118, 
        # 'end_time': 1717195578.8441463, 'start_location': None, 'end_location': None, 'distance': 17400.000000001455, 
        # 'plusview_uri': 'https://bagdb.pluscn.cn:28443/raw/bags/vault46/2024-06-01/plusview/20240601T062436_pdb-l4e-c0010_0.bag.mp4?_x_user=bagdb@plus.ai', 
        # 'camera_view_uri': None, 'plusview_error': None, 'has_online_events': 2, 'events_count': 8, 'events_processing_km': 0.0, 'events_processing_count': 0}
        
        for baginfo in baginfos:
            # 1. filtered by duration
            if args.only_offline and baginfo['bag_source'] != 'offline':
                logger.debug('filtered by source: {}'.format(baginfo['bag_name']))
                continue
            # 2. filtered by distance && speed
            distance = baginfo['distance']
            duration = baginfo['end_time'] - baginfo['start_time']
            if duration < args.clip_duration:
                logger.debug('filtered by duration: {}'.format(baginfo['bag_name']))
                continue
            speed = distance / duration
            if distance < args.min_distance or speed < args.min_speed:
                logger.debug('filtered by distance: {}'.format(baginfo['bag_name']))
                continue
            # 3. filtered by mode percentage
            if float(baginfo['mode']) < args.min_mode:
                logger.debug('filtered by mode: {}'.format(baginfo['bag_name']))
                continue
            # 4. filtered by events count
            if baginfo['events_count'] < args.min_events_count:
                logger.debug('filtered by events count: {}'.format(baginfo['bag_name']))
                continue
            filtered_baginfos.append(baginfo)
        return filtered_baginfos
    
    @staticmethod
    def cluster_bags_to_trip(baginfos, veh_name, thresh=0.1)-> List[TripInfo]:
        trips = []
        bags = []
        for bi in baginfos:
            start_time = float(bi['start_time'])
            end_time = float(bi['end_time'])
            distance = float(bi['distance'])
            speed = distance / (end_time - start_time)
            clip_path ="/mnt" + bi['fastbag_path'].split('/mnt')[1].split('.db')[0] + '.db'
            if bags and start_time - bags[-1].end_time > thresh:
                trips.append(TripInfo(bags=bags, vehicle_name=veh_name))
                bags = []
            bags.append(BagInfo(path=clip_path, start_time=start_time, end_time=end_time, distance=distance, speed=speed))
        return trips
        
    def preprocess_trips(self):
        # divide the trip to multiple clips
        def preprocess(trip: TripInfo):
            trip_save_dir = f'{self.args.output_dir}/{self.args.vehicle_name}/{trip.name()}'
            if os.path.exists(f'{trip_save_dir}/extract_info_done') and args.check_indicator:
                logger.warning(f'Already processed trip: {trip.name()}')
                return
            os.makedirs(trip_save_dir, exist_ok=True)
            os.makedirs(f'{trip_save_dir}/valid_clips', exist_ok=True)
            os.makedirs(f'{trip_save_dir}/invalid_clips', exist_ok=True)
            trip.fragment_clips(self.args.clip_duration)
            trip.download_calib(trip_save_dir)
            with open(f'{trip_save_dir}/bags.txt', 'w') as f:
                for bag in trip.bags:
                    f.write(bag.path + '\n')
            logger.info(f'PreProcess {trip.name()} done')
            
        if self.args.work_nums > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.work_nums) as executor:
                executor.map(preprocess, self.trips)
        else:
            for trip in self.trips:
                preprocess(trip)
        
        indexs = [(i, j) for j in range(len(self.trips)) for i in range(len(self.trips[j].clips))]
        self.indexs = [[i, ind[1], ind[0]] for i, ind in enumerate(indexs)]
        logger.warning(f'found total {len(self.indexs)} clips for vehicle: {self.args.vehicle_name}')
    
    def process_clips(self):
        # download the clips
        start_time = time.time()
        valid = [0 for _ in range(len(self.indexs))]
        def process_unit_clip(index):
            id, ti, ci = index
            save_dir = f'{self.args.output_dir}/{self.args.vehicle_name}/{self.trips[ti].name()}'
            if self.trips[ti].process_clip(ci, save_dir, self.args):
                logger.info(f'Process {id} ({ti}, {ci}) {self.trips[ti].clips[ci].name()} done')
                valid[id] = 1
        
        if self.args.work_nums > 1 and not args.debug_one_data:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.work_nums) as executor:
                executor.map(process_unit_clip, self.indexs)
        else:
            for index in self.indexs:
                process_unit_clip(index)
                if args.debug_one_data:
                    break
        valid_clip_nb = sum(valid)
        cost_time = time.time() - start_time
        ave_time = cost_time / valid_clip_nb if valid_clip_nb > 0 else 0
        logger.warning(f'finished processing {len(self.indexs)} clips, valid clips: {valid_clip_nb}, cost time: {cost_time:.2f}s, average time: {ave_time:.2f}s')
        
        self.attach_finished_indicator()
    
    def attach_finished_indicator(self):
        def attach_info(trip: TripInfo):
            trip_save_dir = f'{self.args.output_dir}/{self.args.vehicle_name}/{trip.name()}'
            if os.path.exists(f'{trip_save_dir}/extract_info_done'):
                return
            os.system(f'touch {trip_save_dir}/extract_info_done')
            
        if self.args.work_nums > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.work_nums) as executor:
                executor.map(attach_info, self.trips)
        else:
            for trip in self.trips:
                attach_info(trip)


def init_logger(args: argparse.Namespace, log_file) -> logging.Logger:
    logger = logging.getLogger(log_file)
    logger.setLevel(getattr(logging, args.min_log_level.upper()))
    logger.propagate = False
    
    formatter = logging.Formatter('%(filename)s:%(lineno)d:%(levelname)s: %(message)s')
    
    # Remove existing handlers to ensure clean setup
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, args.log_level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


# we get candidate bags for auto-labeling, the pipeline is as follows:
# 1. search for all the bags in bag db based vehicle name and date range
# 2. for each bag, we divide it into a set of non-overlapping clips, each clip is n seconds long, which is the smallest unit for auto-labeling
# TODO: for better accuracy, we should propagate the obstacle info from one clip to another, for example, if an obstacle occurs in nearby region
# in clip k, however it's only detected in long distance in clip k+1, we could propagate the obstacle info from clip k to clip k+1.
# 3. for each clip, we evaluate it to determine whether it's a good candidate for auto-labeling, the evaluation criteria is as follows:
# obstacle info, ego info, and other info
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get candidate bags for auto-labeling')
    parser.add_argument('--vehicle_name', type=str, default='pdb-l4e-c0010', help='vehicle name')
    parser.add_argument('--start_date', type=str, default='unknown', help='start date')
    parser.add_argument('--end_date', type=str, default='unknown', help='end date')
    parser.add_argument('--start_time', type=str, default='00:00:00', help='start time')
    parser.add_argument('--end_time', type=str, default='23:59:59', help='end time')
    parser.add_argument('--only_offline', type=bool, default=True, help='only offline')
    parser.add_argument('--min_speed', type=int, default=8, help='min speed(m/s)')
    parser.add_argument('--min_distance', type=int, default=100, help='min distance(m)')
    parser.add_argument('--min_mode', type=float, default=0.5, help='min auto mode percentage')
    parser.add_argument('--min_events_count', type=int, default=0, help='min events')
    parser.add_argument('--clip_duration', type=int, default=10, help='clip duration')
    parser.add_argument('--output_dir', type=str, default='/mnt/bigfile_2/pure_vision_labeling/trips', help='output dir')
    parser.add_argument('--work_nums', type=int, default=4, help='work nums')
    
    # args for chosing clips
    parser.add_argument('--min_hour', type=int, default=0, help='min hour')
    parser.add_argument('--max_hour', type=int, default=24, help='max hour')
    parser.add_argument('--min_side_obs_num', type=int, default=3, help='min obs num')
    parser.add_argument('--min_side_large_obs_num', type=int, default=1, help='min side large obs num')
    parser.add_argument('--use_pitch_correction', type=bool, default=True, help='use pitch correction')
    parser.add_argument('--min_ave_speed', type=float, default=10, help='min ave speed')
    parser.add_argument('--max_ave_speed', type=float, default=100, help='max ave speed')
    parser.add_argument('--mine_non_vehicle_obstacle', type=bool, default=False, help='mine non vehicle obstacle')
    parser.add_argument('--min_cone_frames', type=int, default=0, help='min cone frames')
    parser.add_argument('--min_moto_frames', type=int, default=0, help='min moto frames')
    parser.add_argument('--min_bycycle_frames', type=int, default=0, help='min bycycle frames')
    parser.add_argument('--min_pedestrian_frames', type=int, default=0, help='min pedestrian frames')
    
    # args for roi
    parser.add_argument('--min_x', type=int, default=-80, help='min x')
    parser.add_argument('--max_x', type=int, default=100, help='max x')
    parser.add_argument('--min_y', type=int, default=-8, help='min y')
    parser.add_argument('--max_y', type=int, default=8, help='max y')
    parser.add_argument('--min_side_x', type=int, default=-20, help='min side x')
    parser.add_argument('--max_side_x', type=int, default=20, help='max side x')
    parser.add_argument('--min_side_y', type=int, default=-6, help='min side y')
    parser.add_argument('--max_side_y', type=int, default=6, help='max side y')
    
    parser.add_argument('--min_log_level', type=str, default='DEBUG', help='min log level')
    parser.add_argument('--log_level', type=str, default='INFO', help='log level')
    parser.add_argument('--skip', type=bool, default=False, help='skip')
    parser.add_argument('--debug_mode', type=bool, default=False, help='debug mode')
    parser.add_argument('--debug_one_data', type=bool, default=False, help='debug one data')
    parser.add_argument('--check_indicator', type=bool, default=True, help='check indicator')
    parser.add_argument('--config_yaml', type=str, default=None, help='config yaml')
    
    default_args = parser.parse_args()
    configs = [{}]
    if default_args.config_yaml:
        with open(default_args.config_yaml, 'r') as f:
            configs = yaml.safe_load(f)["trip_fragmentation_configs"]
    for config in configs:
        args = argparse.Namespace()
        for key, value in default_args.__dict__.items():
            setattr(args, key, value)
        for key, value in config.items():
            setattr(args, key, value)
        if args.skip:
            continue
        
        # os.cpu_count() can return None, hence the fallback to 1
        # setattr(args, 'work_nums', min(args.work_nums, (os.cpu_count() or 1) + 1)) # Example heuristic for I/O-bound tasks


        log_file_name = f'{args.output_dir}/{args.vehicle_name}/logs/{args.start_date}_{args.end_date}.log'
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        logger = init_logger(args, log_file_name)
    
        start_time = time.time()
        task_manager = BatchTripTaskManager(args)
        
        task_manager.preprocess_trips()
        task_manager.process_clips()
        end_time = time.time()
        time_elapsed = end_time - start_time
        average_time_per_clip = time_elapsed / len(task_manager.indexs) if len(task_manager.indexs) > 0 else 0
        logger.error(f'finished processing all trips, time cost: {time_elapsed:.2f}s for {len(task_manager.indexs)} clips, average time per clip: {average_time_per_clip:.2f}s')
        
        # finished processing 81 clips, valid clips: 32, cost time: 814.87s, average time: 25.46s
        # finished processing all trips, time cost: 815.24s for 81 clips, average time per clip: 10.06s
