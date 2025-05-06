# /usr/bin/python3
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union, Type, Dict, overload
from datetime import datetime, timezone, timedelta
from enum import IntEnum
import os, sys
import subprocess


class GpuTaskType(IntEnum):
    PRE_LABELING = 0 # pre-labeling is we label with calib-mode proto config
    CALIBRATION = 1 
    VEHICLE_AUTO_LABELING = 2
    PEDESTRIAN_AUTO_LABELING = 3


GpuTaskMemoryDict: Dict[GpuTaskType, int] = {
    GpuTaskType.PRE_LABELING: 3300,
    GpuTaskType.CALIBRATION: 3300,
    GpuTaskType.VEHICLE_AUTO_LABELING: 3300,
    GpuTaskType.PEDESTRIAN_AUTO_LABELING: 3300
}
    

class ClipStatus(IntEnum):
    UNDER_EXTRACTION = 0
    TO_BE_PRELABELED = 1
    BLOCKED_BY_CALIB_REFINEMENT = 2
    TO_BE_AUTO_LABELED = 3
    AUTO_LABELED = 4
    TO_BE_EXTRACTED = 5
    
    
@dataclass
class GpuTask:
    type: GpuTaskType
    process: subprocess.Popen
    memory: int # in MB


def check_path(path):
    if not os.path.exists(path):
        print(f'{path} does not exist')
        sys.exit(1)

def datastr_to_timestamp(datastr):
    dt = datetime.strptime(datastr, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone(timedelta(hours=8)))
    timestamp = dt.timestamp()
    return timestamp

def formatstr_to_timestamp(datastr, format_str):
    dt = datetime.strptime(datastr, format_str).replace(tzinfo=timezone(timedelta(hours=8)))
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
