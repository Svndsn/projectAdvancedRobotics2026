import mujoco
import numpy as np
import pytest

from sensors.camera import Camera  # Assuming your class is in camera.py


def create_dummy_mj_model():
    # Create a minimal MuJoCo model for testing
    xml = """
    <mujoco>
        <worldbody>
            <body name="camera_body">
                <geom type="sphere" size="0.01" />
                <camera name="test_camera" pos="0 0 1" euler="0 0 0" />
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


@pytest.fixture
def camera_instance():
    model, data = create_dummy_mj_model()
    return Camera(model, data, "test_camera")


def test_intrinsics(camera_instance: Camera):
    K = camera_instance.K
    assert K.shape == (3, 3), "Intrinsic matrix should be 3x3"
    assert np.all(np.isfinite(K)), "Intrinsic matrix should have finite values"


def test_extrinsics(camera_instance: Camera):
    T = camera_instance.T_world_cam
    assert T.shape == (4, 4), "Extrinsic matrix should be 4x4"
    assert np.all(np.isfinite(T.A)), "Extrinsic matrix should have finite values"


def test_image_capture(camera_instance: Camera):
    img = camera_instance.image
    assert img is not None, "Image capture should not return None"
    assert isinstance(img, np.ndarray), "Image should be a NumPy array"
