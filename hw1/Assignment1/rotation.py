import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Unnormalized quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        Normalized quaternion with shape (4,)
    """
    return q/np.linalg.norm(q)
    raise NotImplementedError("Implement this function")


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Return the conjugate of the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The conjugate of the quaternion with shape (4,)
    """
    # stupid bug!!
    # q[0]=-q[0]
    # return -q
    return np.array([q[0],-q[1],-q[2],-q[3]])
    raise NotImplementedError("Implement this function")


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    np.ndarray
        The multiplication result with shape (4,)
    """
    w=q1[0]*q2[0]-np.dot(q1[1:],q2[1:])
    v=q1[0]*q2[1:]+q2[0]*q1[1:]+np.cross(q1[1:],q2[1:])
    return np.concatenate(([w], v))
    raise NotImplementedError("Implement this function")


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Use quaternion to rotate a 3D vector.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)
    v: np.ndarray
        Vector with shape (3,)

    Returns
    -------
    np.ndarray
        The rotated vector with shape (3,)
    """
    #q=quat_normalize(q)
    x=np.concatenate(([0], v))
    q_inverse=quat_conjugate(q)
    res=quat_multiply(quat_multiply(q,x),q_inverse)
    return res[1:]
    raise NotImplementedError("Implement this function")


def quat_relative_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute the relative rotation angle between the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    float
        The relative rotation angle in radians, greater than or equal to 0.
    """
    dot=np.dot(q1,q2)
    return 2*np.arccos(abs(dot))
    raise NotImplementedError("Implement this function")


def interpolate_quat(q1: np.ndarray, q2: np.ndarray, ratio: float) -> np.ndarray:
    """
    Interpolate between two quaternions with given ratio.

    When the ratio is 0, return q1; when the ratio is 1, return q2.

    The interpolation should be done in the shortest minor arc connecting the quaternions on the unit sphere.

    If there are multiple correct answers, you can output any of them.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)
    ratio: float
        The ratio of interpolation, should be in [0, 1]

    Returns
    -------
    np.ndarray
        The interpolated quaternion with shape (4,)

    Note
    ----
    What should be done if the inner product of the quaternions is negative?
    """
    epsilon = 1e-3
    dot_product = np.dot(q1, q2)
    # Ensure shortest path by negating q2 if dot product is negative
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    phi = np.arccos(dot_product)
    #phi=quat_relative_angle(q1,q2)/2
    # If angle is very small, use LERP to avoid division by zero
    if phi < epsilon:
        return (1 - ratio) * q1 + ratio * q2
    s1 = np.sin((1 - ratio) * phi) / np.sin(phi)
    s2 = np.sin(ratio * phi) / np.sin(phi)

    return s1 * q1 + s2 * q2
    raise NotImplementedError("Implement this function")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to rotation matrix.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    v_x,v_y,v_z=q[1],q[2],q[3]
    v_mat = np.array([
        [0, -v_z, v_y],
        [v_z, 0, -v_x],
        [-v_y, v_x, 0]
    ])
    E=np.concatenate((-q[1:].reshape(3, 1), (q[0]*np.eye(3)+v_mat)),axis=1)
    G=np.concatenate((-q[1:].reshape(3, 1), (q[0]*np.eye(3)-v_mat)),axis=1)
    return E@G.T
    raise NotImplementedError("Implement this function")


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to quaternion.

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    t=np.trace(mat)+1 #R11+R22+R33=3*w^2-(v_x^2+v_y^2+v_z^2)
    w=np.sqrt(t)/2
    x=(mat[2][1]-mat[1][2])/(4*w)
    z=(mat[1][0]-mat[0][1])/(4*w)
    y=(mat[0][2]-mat[2][0])/(4*w)
    return np.array([w,x,y,z])

    raise NotImplementedError("Implement this function")


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi.

    If there are multiple answers, you can output any.

    Parameters
    ----------
    q: np.ndarray
        The quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    epsilon = 1e-6
    w = q[0]
    x, y, z = q[1:]
    # Handle the case where w is negative
    if w < 0:
        w = -w
        x, y, z = -x, -y, -z
    theta = 2 * np.arccos(w)
    # Handle the case where theta is close to 0 or pi
    if theta < epsilon:
        return np.zeros(3)
    else:
        omega = np.array([x, y, z]) / np.sin(theta / 2)
        return omega * theta
    raise NotImplementedError("Implement this function")


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to quaternion.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    theta = np.linalg.norm(aa)
    if theta < 1e-4:  # Handle small angles
        omega = np.zeros(3)  # No rotation, so axis is undefined or zero vector
    else:
        omega = aa / theta  # Normalize the vector to get the rotation axis
    quat_w=np.cos(theta/2)
    quat_v=np.sin(theta/2)*omega
    return np.concatenate(([quat_w], quat_v))
    raise NotImplementedError("Implement this function")


def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to rotation matrix.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    return quat_to_mat(axis_angle_to_quat(aa))


def mat_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    return quat_to_axis_angle(mat_to_quat(mat))


def uniform_random_quat() -> np.ndarray:
    """
    Generate a random quaternion with uniform distribution.

    Returns
    -------
    np.ndarray
        The random quaternion with shape (4,)
    """

    mean = np.zeros(4)
    covariance_matrix = np.eye(4)
    random_variable = np.random.multivariate_normal(mean, covariance_matrix)
    return quat_normalize(random_variable)
    raise NotImplementedError("Implement this function")


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """
    Convert roll-pitch-yaw euler angles into rotation matrix.

    This is required since URDF use this as rotation representation.

    Parameters
    ----------
    rpy: np.ndarray
        The euler angles with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    roll, pitch, yaw = rpy

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x  # Matrix multiplication in ZYX order
    return R
