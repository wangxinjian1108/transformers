# /usr/bin/python3
import logging
import os, sys
import GPUtil, time
import argparse
from dataclasses import dataclass, field
import glob
from datetime import datetime
import yaml
import json
from tqdm import tqdm
from utils import *
import concurrent.futures

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(filename)s:%(lineno)d:%(levelname)s]:%(message)s -',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


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
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


@dataclass
class ObstacleCounter:
    clip_path: str = ''
    tracking_frames: int = 0
    vehicle: int = 0
    pedestrian: int = 0
    cone: int = 0
    moto: int = 0
    cyclist: int = 0
    

def clip_counter(clip_dir):
    counter = ObstacleCounter()
    clip_info_file = os.path.join(clip_dir, 'clip_info.json')
    if not os.path.exists(clip_info_file):
        return counter
    with open(clip_info_file, 'r') as f:
        clip_info = json.load(f)
    counter.clip_path = clip_dir
    # counter.vehicle = clip_info['vehicle_frames']
    if 'cone_frames' in clip_info:
        counter.tracking_frames = clip_info['tracking_frames']
        counter.pedestrian = clip_info['pedestrian_frames']
        counter.cone = clip_info['cone_frames']
        counter.moto = clip_info['moto_frames']
        counter.cyclist = clip_info['bycycle_frames']
    return counter


def run(args: argparse.Namespace):
    # find all the clips
    total_candidate_clips = []
    st0 = timestamp_to_formatstr(datastr_to_timestamp(args.start_date + ' ' + args.start_time), '%Y%m%dT%H%M%S')
    et0 = timestamp_to_formatstr(datastr_to_timestamp(args.end_date + ' ' + args.end_time), '%Y%m%dT%H%M%S')
    for vehicle_name in tqdm(args.vehicle_names.split(','), desc="Find clips of all vehicle"):
        logger.info(f'Processing vehicle {vehicle_name}')
        veh_dir = os.path.join(args.trip_dir, vehicle_name)
        if not os.path.exists(veh_dir):
            continue
        trips_dir = glob.glob(f'{veh_dir}/{vehicle_name}_*')
        trips_dir.sort()
        logger.debug(f'Found {len(trips_dir)} trips')
        trips_name = [os.path.basename(trip_dir) for trip_dir in trips_dir]
        start_time = [dir.split('_')[1] for dir in trips_name]
        end_time = [dir.split('_')[2] for dir in trips_name]
        time_valids = [st0 <= st < et <= et0 for st, et in zip(start_time, end_time)]
        valid_trips_dir = [dir for dir, tv in zip(trips_dir, time_valids) if tv]
        logger.debug(f'Found {len(valid_trips_dir)} valid trips')
        for trip_dir in tqdm(valid_trips_dir, desc=f'Find valid trips of {vehicle_name}'):
            clips = glob.glob(f'{trip_dir}/valid_clips/*')
            # clips.extend(glob.glob(f'{trip_dir}/invalid_clips/*'))
            logger.debug(f'Found {len(clips)} clips with trip {trip_dir}')
            total_candidate_clips.extend(clips)
    logger.info(f'Found {len(total_candidate_clips)} clips')
    
    pedestrain_valids = ['' for _ in range(len(total_candidate_clips))]
    cone_valids = ['' for _ in range(len(total_candidate_clips))]
    moto_valids = ['' for _ in range(len(total_candidate_clips))]
    cyclist_valids = ['' for _ in range(len(total_candidate_clips))]
    
    def process(index):
        clip_path = total_candidate_clips[index]
        counter = clip_counter(clip_path)
        if counter.tracking_frames > args.max_tracking_frames or counter.cone > args.max_tracking_frames \
            or counter.pedestrian > args.max_tracking_frames or counter.moto > args.max_tracking_frames or \
            counter.cyclist > args.max_tracking_frames:
            logger.warning(f'{clip_path} has too many tracking frames {counter.tracking_frames}, maybe thread unsafe')
            return
        pedestrain_valids[index] = clip_path if counter.pedestrian > args.min_pedestrian_frames else ''
        cone_valids[index] = clip_path if counter.cone > args.min_cone_frames else ''
        moto_valids[index] = clip_path if counter.moto > args.min_moto_frames else ''
        cyclist_valids[index] = clip_path if counter.cyclist > args.min_cyclist_frames else ''
    
    # count the obstacles in multi-thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.work_nums) as executor:
        try:
            list(tqdm(executor.map(process, range(len(total_candidate_clips)), timeout=1000), total=len(total_candidate_clips)))
            # Filter out empty strings and log the results
        except concurrent.futures.TimeoutError:
            logger.error('TimeoutError')
        pedestrain_valids = list(filter(None, pedestrain_valids))
        cone_valids = list(filter(None, cone_valids))
        moto_valids = list(filter(None, moto_valids))
        cyclist_valids = list(filter(None, cyclist_valids))

        logger.warning(f'Found {len(pedestrain_valids)} pedestrian clips')
        logger.warning(f'Found {len(cone_valids)} cone clips')
        logger.warning(f'Found {len(moto_valids)} moto clips')
        logger.warning(f'Found {len(cyclist_valids)} cyclist clips')

        # Save the results to files
        with open(os.path.join(args.save_dir, 'pedestrian_clips.txt'), 'w') as f:
            for item in pedestrain_valids:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, 'cone_clips.txt'), 'w') as f:
            for item in cone_valids:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, 'moto_clips.txt'), 'w') as f:
            for item in moto_valids:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, 'cyclist_clips.txt'), 'w') as f:
            for item in cyclist_valids:
                f.write("%s\n" % item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process clips continuously')
    parser.add_argument('--trip_dir', type=str, default='/mnt/bigfile_2/pure_vision_labeling/trips', help='trip dir')
    parser.add_argument('--vehicle_names', type=str, default='pdb-l4e-b0003', help='vehicle name')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='start date')
    parser.add_argument('--end_date', type=str, default='2024-01-31', help='end date')
    parser.add_argument('--start_time', type=str, default='00:00:00', help='start time')
    parser.add_argument('--end_time', type=str, default='23:59:59', help='end time')
    parser.add_argument('--min_log_level', type=str, default='INFO', help='min log level')
    parser.add_argument('--log_level', type=str, default='INFO', help='log level')
    parser.add_argument('--config_yaml', type=str, default='', help='yaml file')
    parser.add_argument('--max_tracking_frames', type=int, default=200, help='max tracking frames')
    parser.add_argument('--min_pedestrian_frames', type=int, default=100, help='min pedestrian frames')
    parser.add_argument('--min_cone_frames', type=int, default=100, help='min cone frames')
    parser.add_argument('--min_moto_frames', type=int, default=100, help='min moto frames')
    parser.add_argument('--min_cyclist_frames', type=int, default=100, help='min cyclist frames')
    parser.add_argument('--work_nums', type=int, default=1024, help='work nums')
    parser.add_argument('--save_dir', type=str, default='', help='save dir')
    
    args = parser.parse_args()
    
    if args.config_yaml:
        with open(args.config_yaml, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if not hasattr(args, key):
                    continue
                setattr(args, key, value)

    log_file = os.path.join(os.getcwd(), 'mine_clips.log')
    os.system(f'rm -f {log_file}')
    logger = init_logger(args, log_file)
    # check the directory exists, else exit
    if not os.path.exists(args.trip_dir):
        logger.error(f'Trip dir {args.trip_dir} does not exist')
        sys.exit(1)
        
    if not os.path.exists(args.save_dir):
        logger.error(f'Save dir {args.save_dir} does not exist')
        os.makedirs(f'{os.getcwd()}/tmp', exist_ok=True)
        args.save_dir = f'{os.getcwd()}/tmp'
    
    run(args)
