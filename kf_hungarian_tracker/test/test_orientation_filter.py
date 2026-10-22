# Copyright 2026 Firefly Automatix, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A track's orientation is Kalman filtered, not copied from each detection.

Yaw is filtered with a constant yaw-rate model, alongside position. Roll and
pitch are taken from the latest detection. Two properties of angles need care:

- yaw wraps at +-pi, so a box turning through pi must not be smoothed the
  long way round through 0;
- a box turned half a turn about its own z is the same box, so a detector
  that reports either yaw must not make the track spin.
"""

import math

from nav2_dynamic_msgs.msg import Obstacle
import numpy as np
from scipy.spatial.transform import Rotation

from kf_hungarian_tracker.obstacle_class import ObstacleClass

DT = 0.1


def _obstacle(yaw, pitch=0.0, roll=0.0):
    q = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_quat()
    msg = Obstacle()
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
    return msg


def _track(yaw, **kwargs):
    return ObstacleClass(
        _obstacle(yaw),
        top_down=True,
        measurement_noise_cov=[1.0, 1.0, 1.0],
        error_cov_post=[1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
        process_noise_cov=[2.0, 2.0, 0.5],
        **kwargs,
    )


def _step(track, yaw, pitch=0.0, roll=0.0):
    track.predict(DT)
    track.correct(_obstacle(yaw, pitch, roll))


def _rotation(q):
    return Rotation.from_quat([q.x, q.y, q.z, q.w])


def _yaw(q):
    return _rotation(q).as_euler('ZYX')[0]


def _angle_between(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def test_noisy_yaw_is_smoothed():
    """The track's yaw scatters less than the detections it is built from."""
    rng = np.random.default_rng(0)
    truth = 0.5
    measured = truth + rng.normal(0.0, 0.2, 200)
    track = _track(measured[0])
    errors = []
    for yaw in measured[1:]:
        _step(track, yaw)
        errors.append(_angle_between(_yaw(track.msg.orientation), truth))
    tail = np.array(errors[50:])
    raw = np.abs(measured[51:] - truth)
    assert np.sqrt(np.mean(tail**2)) < 0.5 * np.sqrt(np.mean(raw**2))


def test_yaw_rate_is_estimated_and_coasted():
    """A box turning steadily keeps turning while it is missed."""
    rate = 0.5  # rad/s
    track = _track(0.0)
    for i in range(1, 60):
        _step(track, rate * DT * i)
    assert abs(float(track.kalman.statePost[7][0]) - rate) < 0.05

    last = _yaw(track.msg.orientation)
    for _ in range(5):
        track.predict(DT)
    coasted = _yaw(track.msg.orientation)
    assert _angle_between(coasted, last + 5 * DT * rate) < 0.05


def test_turning_through_pi_does_not_go_the_long_way():
    """Yaw goes 170, 175, 180, -175 ... deg; the track follows, never via 0."""
    start = math.radians(160.0)
    rate = math.radians(50.0)  # rad/s, 5 deg per frame
    track = _track(start)
    for i in range(1, 30):
        truth = start + rate * DT * i
        _step(track, math.atan2(math.sin(truth), math.cos(truth)))
        off = _angle_between(_yaw(track.msg.orientation), truth)
        assert off < math.radians(10.0), (
            f'frame {i}: true yaw {math.degrees(truth):.0f} deg, track at '
            f'{math.degrees(_yaw(track.msg.orientation)):.0f} deg'
        )


def test_half_turn_ambiguity_does_not_spin_the_track():
    """A detector alternating between yaw and yaw + pi keeps the track still."""
    yaw = math.radians(30.0)
    track = _track(yaw)
    for i in range(20):
        _step(track, yaw + (math.pi if i % 2 else 0.0))
        assert _angle_between(_yaw(track.msg.orientation), yaw) < 1e-3
    assert abs(float(track.kalman.statePost[7][0])) < 1e-3


def test_flipped_detection_describes_the_same_box():
    """With roll and pitch, the flipped yaw still gives the detected box."""
    yaw, pitch, roll = math.radians(30.0), 0.1, -0.05
    track = _track(yaw)
    for _ in range(20):
        _step(track, yaw, pitch, roll)
    _step(track, yaw + math.pi, pitch, roll)

    detected = Rotation.from_euler('ZYX', [yaw + math.pi, pitch, roll])
    tracked = _rotation(track.msg.orientation)
    # The box's own axes, up to a half turn about its z.
    half_turn = Rotation.from_euler('z', math.pi)
    candidates = [detected, detected * half_turn]
    assert min((tracked.inv() * c).magnitude() for c in candidates) < 1e-3
