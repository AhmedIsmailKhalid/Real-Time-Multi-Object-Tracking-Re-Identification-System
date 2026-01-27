"""
Kalman filter for object motion prediction.
Predicts object position in next frame based on motion history.
"""

import numpy as np


class KalmanFilter:
    """
    Kalman filter for bounding box tracking.

    State vector: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
    - x_center, y_center: Center coordinates
    - aspect_ratio: width / height
    - height: Box height
    - vx, vy, va, vh: Velocities of the above

    Measurement vector: [x_center, y_center, aspect_ratio, height]
    """

    def __init__(self):
        """Initialize Kalman filter with constant velocity model."""
        ndim = 4  # x, y, a, h
        dt = 1.0  # Time step

        # State transition matrix (constant velocity model)
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # Measurement matrix (only observe position, not velocity)
        self.H = np.eye(ndim, 2 * ndim)

        # Process noise covariance
        self.Q = np.eye(2 * ndim)
        self.Q[ndim:, ndim:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance
        self.R = np.eye(ndim)
        self.R[2:, 2:] *= 10.0  # Higher noise for aspect ratio and height

        # State covariance
        self.P = np.eye(2 * ndim)
        self.P[ndim:, ndim:] *= 1000.0  # High uncertainty for initial velocities

        # State vector
        self.x = np.zeros((2 * ndim, 1))

    def predict(self):
        """
        Predict next state.

        Returns:
            Predicted state vector
        """
        # x = F * x
        self.x = np.dot(self.F, self.x)

        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x

    def update(self, measurement: np.ndarray):
        """
        Update state with measurement.

        Args:
            measurement: Measurement vector [x, y, a, h]
        """
        measurement = measurement.reshape((4, 1))

        # y = z - H * x (innovation)
        y = measurement - np.dot(self.H, self.x)

        # S = H * P * H^T + R (innovation covariance)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # K = P * H^T * S^-1 (Kalman gain)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # x = x + K * y
        self.x = self.x + np.dot(K, y)

        # P = (I - K * H) * P
        I = np.eye(self.P.shape[0])  # noqa: E741
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self) -> np.ndarray:
        """
        Get current state.

        Returns:
            State vector [x, y, a, h, vx, vy, va, vh]
        """
        return self.x.flatten()


def bbox_to_z(bbox: tuple[float, float, float, float]) -> np.ndarray:
    """
    Convert bounding box to measurement vector.

    Args:
        bbox: Bounding box (x1, y1, x2, y2)

    Returns:
        Measurement vector [x_center, y_center, aspect_ratio, height]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    a = w / h if h > 0 else 1.0
    return np.array([x, y, a, h])


def z_to_bbox(z: np.ndarray) -> tuple[float, float, float, float]:
    """
    Convert measurement vector to bounding box.

    Args:
        z: Measurement vector [x_center, y_center, aspect_ratio, height]

    Returns:
        Bounding box (x1, y1, x2, y2)
    """
    x, y, a, h = z[:4]
    w = a * h
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return (x1, y1, x2, y2)
