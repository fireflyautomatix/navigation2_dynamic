import numpy as np
import cv2
import uuid
from scipy.spatial.transform import Rotation
from unique_identifier_msgs.msg import UUID


def wrap(angle, period=2 * np.pi):
    """angle shifted by whole periods into [-period/2, period/2)"""
    return (angle + period / 2) % period - period / 2


class ObstacleClass:
    """wrap a kalman filter and extra information for one single obstacle

    State space is 3D (x, y, z) by default, if you want to work on 2D (for example top-down view), you can simply make z a constant value and independent of x, y.

    Arrtibutes:
        position: 3d position of center point, numpy array with shape (3, 1)
        velocity: 3d velocity of center point, numpy array with shape (3, 1)
        kalman: cv2.KalmanFilter on (x, y, z, vx, vy, vz, yaw, yaw rate);
            roll and pitch follow the latest detection
        dying: count missing frames for this obstacle, if reach threshold, delete this obstacle
    """

    def __init__(
        self,
        obstacle_msg,
        top_down,
        measurement_noise_cov,
        error_cov_post,
        process_noise_cov,
        yaw_measurement_noise_cov=0.1,
        yaw_error_cov_post=(0.1, 1.0),
        yaw_process_noise_cov=1.0,
    ):
        """Initialize with an Obstacle msg and an assigned id"""
        self.msg = obstacle_msg

        uuid_msg = UUID()
        uuid_msg.uuid = list(uuid.uuid4().bytes)
        self.msg.uuid = uuid_msg

        position = np.array(
            [
                [
                    obstacle_msg.position.x,
                    obstacle_msg.position.y,
                    obstacle_msg.position.z,
                ]
            ]
        ).T  # shape 3*1
        velocity = np.array(
            [
                [
                    obstacle_msg.velocity.x,
                    obstacle_msg.velocity.y,
                    obstacle_msg.velocity.z,
                ]
            ]
        ).T

        # check aganist state space dimension, top_down or not
        if top_down:
            measurement_noise_cov[2] = 0.0
            error_cov_post[2] = 0.0
            error_cov_post[5] = 0.0
            process_noise_cov[2] = 0.0

        # setup kalman filter: 8d state (x, y, z, vx, vy, vz, yaw, yaw rate)
        # and 4d observation (x, y, z, yaw)
        yaw, self.pitch, self.roll = self._euler(obstacle_msg.orientation)
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            np.float32,
        )
        self.kalman.measurementNoiseCov = np.diag(
            list(measurement_noise_cov) + [yaw_measurement_noise_cov]
        ).astype(np.float32)
        self.kalman.statePost = np.concatenate(
            [position, velocity, [[yaw], [0.0]]]
        ).astype(np.float32)
        self.kalman.errorCovPost = np.diag(
            list(error_cov_post) + list(yaw_error_cov_post)
        ).astype(np.float32)

        self.dying = 0
        self.top_down = top_down
        self.process_noise_cov = process_noise_cov
        self.yaw_process_noise_cov = yaw_process_noise_cov
        self._publish_orientation(self.kalman.statePost)

    @staticmethod
    def _euler(q):
        """(yaw, pitch, roll) of a quaternion msg, intrinsic z-y'-x''"""
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("ZYX")

    def _publish_orientation(self, state):
        """write the filtered yaw, with the latest detection's roll and pitch, into msg"""
        q = Rotation.from_euler(
            "ZYX", [float(state[6][0]), self.pitch, self.roll]
        ).as_quat()
        self.msg.orientation.x = float(q[0])
        self.msg.orientation.y = float(q[1])
        self.msg.orientation.z = float(q[2])
        self.msg.orientation.w = float(q[3])

    def predict(self, dt):
        """update F and Q matrices, call KalmanFilter.predict and store position and velocity"""

        # construct new transition matrix
        """
        F = 1, 0, 0, dt, 0,  0,  0, 0
            0, 1, 0, 0,  dt, 0,  0, 0
            0, 0, 1, 0,  0,  dt, 0, 0
            0, 0, 0, 1,  0,  0,  0, 0
            0, 0, 0, 0,  1,  0,  0, 0
            0, 0, 0, 0,  0,  1,  0, 0
            0, 0, 0, 0,  0,  0,  1, dt
            0, 0, 0, 0,  0,  0,  0, 1
        """
        F = np.eye(8).astype(np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[6, 7] = dt

        # construct new process conv matrix
        """assume constant velocity and yaw rate, with acceleration noise ax, ay, az in x, y, z
        and angular acceleration noise aw in yaw; each (value, rate) pair gets the block
            dt4*a/4, dt3*a/2
            dt3*a/2, dt2*a
        e.g. for x, Q[0, 0] = dt4*ax/4, Q[0, 3] = Q[3, 0] = dt3*ax/2, Q[3, 3] = dt2*ax
        """
        dt2 = dt**2
        dt3 = dt * dt2
        dt4 = dt2**2

        Q = np.zeros((8, 8), np.float32)
        noise = list(self.process_noise_cov) + [self.yaw_process_noise_cov]
        for value, rate, a in zip((0, 1, 2, 6), (3, 4, 5, 7), noise):
            Q[value, value] = dt4 * a / 4
            Q[value, rate] = Q[rate, value] = dt3 * a / 2
            Q[rate, rate] = dt2 * a

        self.kalman.transitionMatrix = F
        self.kalman.processNoiseCov = Q
        self.kalman.predict()
        self.msg.position.x = float(self.kalman.statePre[0][0])
        self.msg.position.y = float(self.kalman.statePre[1][0])
        self.msg.position.z = float(self.kalman.statePre[2][0])
        self.msg.velocity.x = float(self.kalman.statePre[3][0])
        self.msg.velocity.y = float(self.kalman.statePre[4][0])
        self.msg.velocity.z = float(self.kalman.statePre[5][0])

        # predict copies statePre into statePost before we wrap, so wrap both
        self.kalman.statePre[6][0] = wrap(self.kalman.statePre[6][0])
        self.kalman.statePost[6][0] = self.kalman.statePre[6][0]
        self._publish_orientation(self.kalman.statePre)

    def correct(self, detect_msg):
        """extract position and yaw as measurement and update KalmanFilter"""
        if self.top_down:
            detect_msg.position.z = 0.0

        # A box turned half a turn about its own z is the same box, so the
        # detector may report either yaw. Measure the one nearest the
        # prediction, and unwrap it so the innovation never jumps by 2*pi.
        # Rz(y + pi) Ry(-p) Rx(-r) == Rz(y) Ry(p) Rx(r) Rz(pi), so the
        # flipped yaw comes with roll and pitch negated.
        yaw, pitch, roll = self._euler(detect_msg.orientation)
        predicted = float(self.kalman.statePost[6][0])
        innovation = wrap(yaw - predicted)
        if abs(innovation) > np.pi / 2:
            innovation = wrap(innovation, np.pi)
            pitch, roll = -pitch, -roll
        self.pitch, self.roll = pitch, roll

        measurement = np.array(
            [
                [
                    detect_msg.position.x,
                    detect_msg.position.y,
                    detect_msg.position.z,
                    predicted + innovation,
                ]
            ]
        ).T.astype(np.float32)
        self.kalman.correct(measurement)
        self.kalman.statePost[6][0] = wrap(self.kalman.statePost[6][0])
        self.msg.position.x = float(self.kalman.statePost[0][0])
        self.msg.position.y = float(self.kalman.statePost[1][0])
        self.msg.position.z = float(self.kalman.statePost[2][0])
        self.msg.velocity.x = float(self.kalman.statePost[3][0])
        self.msg.velocity.y = float(self.kalman.statePost[4][0])
        self.msg.velocity.z = float(self.kalman.statePost[5][0])
        self.msg.size = detect_msg.size
        self._publish_orientation(self.kalman.statePost)
        self.msg.position_covariance = detect_msg.position_covariance
        self.msg.velocity_covariance = detect_msg.velocity_covariance

    def distance(self, other_msg):
        """measurement distance between two obstacles, dy default it's Euler distance between centers
        you can extent the Obstacle msg to include more features like class or uncertainty and include in the distance function"""
        position = np.array(
            [[self.msg.position.x, self.msg.position.y, self.msg.position.z]]
        ).T
        other_position = np.array(
            [[other_msg.position.x, other_msg.position.y, other_msg.position.z]]
        ).T
        return np.linalg.norm(position - other_position)
