"""This module provides classes for working with transformations in 3D space."""

from typing import Annotated, Optional, Self, TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation as R

# Frame = 4x4 transformation matrix
TransformationMatrix: TypeAlias = Annotated[np.ndarray, "Shape(4, 4)", "dtype(float)"]
Position: TypeAlias = Annotated[np.ndarray, "Shape(3,)", "dtype(float)"]
Quaternion: TypeAlias = Annotated[np.ndarray, "Shape(4,)", "dtype(float)"]
Euler: TypeAlias = Annotated[np.ndarray, "Shape(3,)", "dtype(float)"]


class Frame:
    def __init__(self, mat: TransformationMatrix, *, degrees: bool, right_handed: bool):
        self._mat = mat
        self._degrees = degrees
        self._right_handed = right_handed

    @classmethod
    def from_frame(cls, frame: Self) -> Self:
        return cls(frame.mat, degrees=frame.degrees, right_handed=frame.right_handed)

    @classmethod
    def create(
        cls,
        *,
        pos: Optional[Position] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        quat: Optional[Quaternion] = None,
        euler: Optional[Euler] = None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        degrees: bool = True,
        right_handed: bool = True,
    ) -> Self:
        """Creates a 4x4 transformation matrix from the given position and orientation.

        NOTE: The order of precedence is euler angles, quaternion, and then roll, pitch,
        yaw. As in, euler angles will be used if provided, otherwise quaternion, and
        then roll, pitch, yaw. Roll, pitch, yaw is applied RPY.

        Keyword Arguments:
            pos (Optional[Position]): The position of the frame. Takes precedence over
                the x, y, z arguments.
            x (Optional[float]): The x position of the frame.
            y (Optional[float]): The y position of the frame.
            z (Optional[float]): The z position of the frame.
            quat (Optional[Quaternion]): The quaternion of the frame. Assumes scalar
                last, so [x, y, z, w].
            euler (Optional[Euler]): The euler angles of the frame.
            roll (Optional[float]): The roll of the frame.
            pitch (Optional[float]): The pitch of the frame.
            yaw (Optional[float]): The yaw of the frame.
            degrees (bool): If True, the euler angles are in degrees. Otherwise,
                they are in radians.
            right_handed (bool): If True, the euler angles are in right handed
                coordinates. Otherwise, they are in left handed coordinates.

        Returns:
            Frame: The 4x4 transformation matrix.
        """
        euler_seq = "xyz" if right_handed else "zyx"

        mat = np.eye(4)

        if pos is not None:
            mat[:3, 3] = pos
        elif x is not None or y is not None or z is not None:
            mat[:3, 3] = np.array([x or 0, y or 0, z or 0])

        if euler is not None:
            mat[:3, :3] = R.from_euler(euler_seq, euler, degrees=degrees).as_matrix()
        elif quat is not None:
            mat[:3, :3] = R.from_quat(quat).as_matrix()
        elif roll is not None or pitch is not None or yaw is not None:
            roll = roll or 0
            pitch = pitch or 0
            yaw = yaw or 0
            angles = [roll, pitch, yaw]
            mat[:3, :3] = R.from_euler(euler_seq, angles, degrees=degrees).as_matrix()

        return cls(mat, degrees=degrees, right_handed=right_handed)

    def apply(self, other: Self, *, T: Optional[Self] = None) -> Self:
        """Applies the given action to the frame and returns a new frame.

        Keyword Arguments:
            T (Optional[TransformationMatrix]): An optional transformation matrix to
                apply to the frame before applying the action.
        """
        T = T if T is not None else other.copy(mat=np.eye(4))
        return self @ T.inverse() @ other @ T

    def apply_inverse(self, other: Self) -> Self:
        """Applies the inverse of the given action to the frame and returns a new
        frame."""
        return self @ other.inverse()

    def inverse(self) -> Self:
        """Returns the inverse of the frame."""
        return self.copy(mat=np.linalg.inv(self._mat))

    def copy(
        self,
        *,
        mat: Optional[TransformationMatrix] = None,
        degrees: Optional[bool] = None,
        right_handed: Optional[bool] = None,
    ) -> Self:
        """Returns a copy of the frame."""
        mat = mat if mat is not None else self._mat
        degrees = degrees if degrees is not None else self._degrees
        right_handed = right_handed if right_handed is not None else self._right_handed
        return type(self)(mat.copy(), degrees=degrees, right_handed=right_handed)

    # ==========================

    @property
    def pos(self) -> Position:
        return self._mat[:3, 3]

    @pos.setter
    def pos(self, new_pos: Position):
        self._mat[:3, 3] = new_pos

    @property
    def quat(self) -> Quaternion:
        return R.from_matrix(self._mat[:3, :3]).as_quat()

    @property
    def euler(self) -> Euler:
        return R.from_matrix(self._mat[:3, :3]).as_euler("xyz", degrees=True)

    @property
    def mat(self) -> TransformationMatrix:
        return self._mat

    @property
    def degrees(self) -> bool:
        return self._degrees

    @property
    def right_handed(self) -> bool:
        return self._right_handed

    # ==========================

    def __repr__(self) -> str:
        return f"Frame(pos={self.pos}, euler={self.euler})"

    def __matmul__(self, other: Self) -> Self:
        return self.copy(mat=self._mat @ other.mat)

    def __mul__(self, scalar: float | int) -> Self:
        assert isinstance(scalar, (float, int)), "Can only multiply by a scalar."
        mat = self._mat.copy()
        mat[:3, 3] *= scalar  # Scale the position
        return self.copy(mat=mat)

    def __rmul__(self, scalar: float | int) -> Self:
        return self * scalar

    def __truediv__(self, scalar: float | int) -> Self:
        return self * (1 / scalar)

    def __rtruediv__(self, scalar: float | int) -> Self:
        return self / scalar

    def __neg__(self) -> Self:
        return self * -1


class Action(Frame):
    def __init__(self, mat: TransformationMatrix, *, degrees: bool, right_handed: bool):
        super().__init__(mat, degrees=degrees, right_handed=right_handed)

    def get(self) -> Annotated[np.ndarray, "Shape(6,)", "dtype(float)"]:
        """Returns the action as a 6 element array: [x, y, z, roll, pitch, yaw]"""
        return np.concatenate([self.pos, self.euler])

    def __repr__(self) -> str:
        return f"Action(pos={self.pos}, euler={self.euler})"


GlobalFrame: TypeAlias = Frame
LocalFrame: TypeAlias = Frame
