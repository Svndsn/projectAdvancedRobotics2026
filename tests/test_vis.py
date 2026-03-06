import numpy as np
import open3d as o3d
import pytest
from _pytest.compat import LEGACY_PATH

from utils.vis import dehom, hom, normalize, pcread, pcwrite


def test_pcwrite_pcread(tmpdir: LEGACY_PATH):
    # Create a sample point cloud as numpy array
    point_cloud_np = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    # Write to a temporary file
    file_path = tmpdir.join("test.pcd")
    pcwrite(str(file_path), point_cloud_np)

    # Read back the point cloud
    loaded_point_cloud = pcread(str(file_path))

    # Check that the loaded point cloud matches the original
    assert np.allclose(point_cloud_np, loaded_point_cloud)

    # Test that pcread raises an error with an invalid file
    with pytest.raises(ValueError):
        pcread("non_existent_file.pcd")


def test_hom():
    # Test with 3x3 input (nx3 format)
    points_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_homogeneous = np.array([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    assert np.array_equal(hom(points_3x3), expected_homogeneous)
    print("a")

    # Test with 3xN input (3xn format)
    points_3xn = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 2, 3]])
    expected_homogeneous_3xn = np.array(
        [[1, 4, 7, 1], [2, 5, 8, 1], [3, 6, 9, 1], [1, 2, 3, 1]]
    )
    assert np.array_equal(hom(points_3xn), expected_homogeneous_3xn)

    # Test with input that's not a numpy array
    points_invalid_type = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    with pytest.raises(ValueError):
        hom(points_invalid_type)


def test_dehom():
    # Valid input case: 4xN
    points_4xN = np.array([[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]])
    expected_output_3xN = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.allclose(dehom(points_4xN), expected_output_3xN)

    # Valid input case: Nx4
    points_Nx4 = np.array([[1, 4, 7, 1], [2, 5, 8, 1], [3, 6, 9, 1]]).T
    assert np.allclose(dehom(points_Nx4), expected_output_3xN)

    # Invalid input case: not a numpy array
    invalid_points_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"{invalid_points_list=}")
    with pytest.raises(ValueError):
        dehom(invalid_points_list)


def test_normalize(tmpdir: LEGACY_PATH):
    # Test with numpy array
    image_array = np.array([[0, 50, 100], [150, 200, 255]])
    normalized_array = normalize(image_array)
    assert np.min(normalized_array) == 0.0
    assert np.max(normalized_array) == 1.0

    # Test with list
    image_list = [[0, 50, 100], [150, 200, 255]]
    normalized_list = normalize(image_list)
    assert np.min(normalized_list) == 0.0
    assert np.max(normalized_list) == 1.0

    # Test with Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
    normalized_pcd = normalize(pcd)
    assert np.min(normalized_pcd) == 0.0
    assert np.max(normalized_pcd) == 1.0

    # Test with an image file (requires an image to be available)
    image_path = tmpdir.join("test_image.png")
    np.random.seed(0)
    image_data = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    from PIL import Image

    Image.fromarray(image_data).save(str(image_path))
    normalized_image = normalize(str(image_path))
    assert np.min(normalized_image) == 0.0
    assert np.max(normalized_image) == 1.0

    # Test with unsupported type
    with pytest.raises(ValueError):
        normalize(12345)  # Should raise ValueError
