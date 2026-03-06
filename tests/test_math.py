import numpy as np
import spatialmath.base as smb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3

from utils.math import (  # replace 'your_module' with the actual name of your module
    angular_distance,
    arbitrary_orthogonal_vector,
    calculate_rotation_between_vectors,
    cint,
    conj,
    dotproduct,
    euclidean_distance,
    flip,
    frobenius_norm,
    gcd,
    geodesic_distance,
    hausdorff_distance,
    homotopy_class,
    lcm,
    length,
    normalize_vector,
    npq2np,
    quat_to_axang,
    random_unit_quaternion,
    rotate_vector_2d,
)


# Test rotate_vector_2d
def test_rotate_vector_2d():
    vector = (1, 0)
    angle = np.pi / 2  # 90 degrees
    rotated_vector = rotate_vector_2d(vector, angle)
    expected_vector = np.array([0, 1])
    assert np.allclose(rotated_vector, expected_vector), (
        f"Expected {expected_vector}, but got {rotated_vector}"
    )


# Test dotproduct
def test_dotproduct():
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    result = dotproduct(v1, v2)
    expected_result = 32
    assert np.isclose(result, expected_result), (
        f"Expected {expected_result}, but got {result}"
    )


# Test length
def test_length():
    v = [3, 4]
    result = length(v)
    expected_result = 5
    assert np.isclose(result, expected_result), (
        f"Expected {expected_result}, but got {result}"
    )


# Test angle
# def test_angle():
#     v1 = [1, 0]
#     v2 = [0, 1]
#     result = angle(v1, v2)
#     expected_result = np.pi / 2
#     assert np.isclose(
#         result, expected_result
#     ), f"Expected {expected_result}, but got {result}"


# Test flip
def test_flip():
    v = [1, -2, 3]
    flipped = flip(v)
    expected_flipped = np.array([-1, 2, -3])
    assert np.allclose(flipped, expected_flipped), (
        f"Expected {expected_flipped}, but got {flipped}"
    )


# Test gcd
def test_gcd():
    result = gcd(48, 18)
    expected_result = 6
    assert np.isclose(result, expected_result), (
        f"Expected {expected_result}, but got {result}"
    )


# Test lcm
def test_lcm():
    result = lcm(4, 6)
    expected_result = 12
    assert np.isclose(result, expected_result), (
        f"Expected {expected_result}, but got {result}"
    )

    result_list = lcm([4, 6, 8])
    expected_result_list = 24
    assert np.isclose(result_list, expected_result_list), (
        f"Expected {expected_result_list}, but got {result_list}"
    )


# Test random_unit_quaternion
def test_random_unit_quaternion():
    q = random_unit_quaternion()
    assert np.isclose(np.linalg.norm(q.A), 1.0), "Quaternion is not a unit quaternion"


# Test normalize_vector
def test_normalize_vector():
    v = np.array([3, 4])
    normalized = normalize_vector(v)
    expected_normalized = np.array([0.6, 0.8])
    assert np.allclose(normalized, expected_normalized), (
        f"Expected {expected_normalized}, but got {normalized}"
    )


# Test calculate_rotation_between_vectors
def test_calculate_rotation_between_vectors():
    v_from = np.array([1, 0, 0])
    v_to = np.array([0, 1, 0])
    rotation_matrix = calculate_rotation_between_vectors(v_from, v_to)
    expected_rotation_matrix = R.from_rotvec(
        np.pi / 2 * np.array([0, 0, 1])
    ).as_matrix()
    assert np.allclose(rotation_matrix, expected_rotation_matrix), (
        "Rotation matrix does not match the expected result"
    )


# Test arbitrary_orthogonal_vector
def test_arbitrary_orthogonal_vector():
    vec = np.array([1, 0, 0])
    ortho_vec = arbitrary_orthogonal_vector(vec)
    assert np.allclose(np.dot(vec, ortho_vec), 0), "Orthogonal vector is not orthogonal"


# Test quat_to_axang
def test_quat_to_axang():
    q = smb.r2q(smb.eul2r([0, 0, np.pi / 2]))
    axang = quat_to_axang(q)
    expected_axang = np.array([[0, 0, 1, np.pi / 2]])
    assert np.allclose(axang, expected_axang), (
        f"Expected {expected_axang}, but got {axang}"
    )


# Test conj
def test_conj():
    q = np.array([1, 2, 3, 4])
    conj_q = conj(q)
    expected_conj_q = np.array([1, -2, -3, -4])
    assert np.allclose(conj_q, expected_conj_q), (
        f"Expected {expected_conj_q}, but got {conj_q}"
    )


# Test npq2np
def test_npq2np():
    npq = np.quaternion(1, 2, 3, 4)
    array = npq2np(npq)
    expected_array = np.array([1, 2, 3, 4])
    assert np.allclose(array, expected_array), (
        f"Expected {expected_array}, but got {array}"
    )


# Test euclidean_distance
def test_euclidean_distance():
    T1 = SE3.Trans([1, 2, 3])
    T2 = SE3.Trans([4, 5, 6])
    dist = euclidean_distance(T1, T2)
    expected_dist = np.linalg.norm(T1.t - T2.t)
    assert np.isclose(dist, expected_dist), f"Expected {expected_dist}, but got {dist}"


# Test angular_distance
def test_angular_distance():
    T1 = SE3.Rz(np.pi / 4)
    T2 = SE3.Rz(-np.pi / 4)
    dist = angular_distance(T1, T2)
    expected_dist = np.pi / 2
    assert np.isclose(dist, expected_dist), f"Expected {expected_dist}, but got {dist}"


# Test frobenius_norm
def test_frobenius_norm():
    T1 = SE3.Rz(np.pi / 4) * SE3.Trans([1, 2, 3])
    T2 = SE3.Rz(-np.pi / 4) * SE3.Trans([4, 5, 6])
    norm = frobenius_norm(T1, T2)
    expected_norm = np.linalg.norm(T1.A - T2.A, "fro")
    assert np.isclose(norm, expected_norm), f"Expected {expected_norm}, but got {norm}"


# Test geodesic_distance
def test_geodesic_distance():
    T1 = SE3.Rz(np.pi / 4) @ SE3.Trans([1, 2, 3])
    T2 = SE3.Rz(-np.pi / 4) @ SE3.Trans([4, 5, 6])
    dist = geodesic_distance(T1, T2)
    T_diff = T1.inv() * T2
    log_map = SE3.log(T_diff)
    expected_dist = np.linalg.norm(log_map)
    assert np.isclose(dist, expected_dist), f"Expected {expected_dist}, but got {dist}"


# Test hausdorff_distance
def test_hausdorff_distance():
    T1 = SE3.Rz(np.pi / 4) * SE3.Trans([1, 2, 3])
    T2 = SE3.Rz(-np.pi / 4) * SE3.Trans([4, 5, 6])
    points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    dist = hausdorff_distance(T1, T2, points)
    transformed_points_T1 = np.dot(T1.A[:3, :3], points.T).T + T1.t
    transformed_points_T2 = np.dot(T2.A[:3, :3], points.T).T + T2.t

    dist_matrix = np.linalg.norm(
        transformed_points_T1[:, np.newaxis, :]
        - transformed_points_T2[np.newaxis, :, :],
        axis=2,
    )
    expected_hausdorff_dist = max(
        np.max(np.min(dist_matrix, axis=1)), np.max(np.min(dist_matrix, axis=0))
    )
    assert np.isclose(dist, expected_hausdorff_dist), (
        f"Expected {expected_hausdorff_dist}, but got {dist}"
    )


def test_homotopy_class():
    def F(z: complex) -> complex:
        return 1 / z

    def generate_circle_points(radius: float, num_points: int) -> list:
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = [(radius * np.cos(theta), radius * np.sin(theta)) for theta in angles]
        return points

    pts = generate_circle_points(1, 100000)

    I, I_e = homotopy_class(F, pts)

    assert np.isclose(I, complex(real=0, imag=2 * np.pi))


def test_cint():
    I, I_e = cint(
        lambda z: 1 / z,
        lambda t: np.exp(1j * t),
        lambda t: 1j * np.exp(1j * t),
        0,
        2 * np.pi,
    )
    assert np.isclose(I, complex(real=0, imag=2 * np.pi))
