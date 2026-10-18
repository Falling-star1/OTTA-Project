"""Explicit physical selection; logical CUDA devices follow the supplied order."""
import re
from job_state import require


def physical_devices(value, expected_count):
    require(type(expected_count) is int and expected_count >= 2, 'Invalid GPU count')
    require(bool(re.fullmatch(r'\d+(,\d+)+', value)), 'Specify comma-separated physical GPU IDs')
    devices = [int(item) for item in value.split(',')]
    require(len(devices) == expected_count and len(set(devices)) == expected_count,
            'Expected ' + str(expected_count) + ' distinct physical GPU IDs')
    return devices


def logical_devices(count):
    return {'cuda:' + str(index) for index in range(count)}
