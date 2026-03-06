import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import glfw
import mujoco as mj
import numpy as np
import spatialmath as sm
from dm_control import mjcf

from sims.base_sim import BaseSim
from utils.mj import ObjType, get_pose, name2id
from utils.vis import pcwrite


class Camera:
    def __init__(
        self,
        model,
        data,
        cam_name: str = "cam",
        save_dir="data/img/",
        width: int = 640,
        height: int = 480,
    ):
        """Initialize Camera instance.

        Args:
        - model: Mujoco model.
        - data: Mujoco data.
        - cam_name: Name of the camera.
        - save_dir: Directory to save captured images.
        """
        self._cam_name = cam_name
        self._model = model
        self._data = data
        self._save_dir = save_dir + self._cam_name + "/"

        self._width = width
        self._height = height
        self._cam_id = name2id(self._model, self._cam_name, obj_type=ObjType.CAMERA)

        self._renderer = mj.Renderer(self._model, self._height, self._width)
        self._camera = mj.MjvCamera()
        self._scene = mj.MjvScene(self._model, maxgeom=10_000)

        self._image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._depth_image = np.zeros((self._height, self._width, 1), dtype=np.float32)
        self._seg_id_image = np.zeros((self._height, self._width, 3), dtype=np.float32)
        self._point_cloud = np.zeros((self._height, self._width, 1), dtype=np.float32)

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    @property
    def height(self) -> int:
        """
        Get the height of the camera.

        Returns:
                int: The height of the camera.
        """
        return self._height

    @property
    def width(self) -> int:
        """
        Get the width of the camera.

        Returns:
                int: The width of the camera.
        """
        return self._width

    @property
    def save_dir(self) -> str:
        """
        Get the directory where images captured by the camera are saved.

        Returns:
                str: The directory where images captured by the camera are saved.
        """
        return self._save_dir

    @property
    def name(self) -> str:
        """
        Get the name of the camera.

        Returns:
                str: The name of the camera.s
        """
        return self._cam_name

    @property
    def K(self) -> np.ndarray:
        """
        Compute the intrinsic camera matrix (K) based on the camera's field of view (fov),
        width (_width), and height (_height) parameters, following the pinhole camera model.

        Returns:
        np.ndarray: The intrinsic camera matrix (K), a 3x3 array representing the camera's intrinsic parameters.
        """
        # Convert the field of view from degrees to radians
        theta = np.deg2rad(self.fov)

        # Focal length calculation (f in terms of sensor width and height)
        f_x = (self._width / 2) / np.tan(theta / 2)
        f_y = (self._height / 2) / np.tan(theta / 2)

        # Pixel resolution (assumed to be focal length per pixel unit)
        alpha_u = f_x  # focal length in terms of pixel width
        alpha_v = f_y  # focal length in terms of pixel height

        # Optical center offsets (assuming they are at the center of the sensor)
        u_0 = (self._width - 1) / 2.0
        v_0 = (self._height - 1) / 2.0

        # Intrinsic camera matrix K
        K = np.array([[alpha_u, 0, u_0], [0, alpha_v, v_0], [0, 0, 1]])

        return K

    @property
    def T_world_cam(self) -> sm.SE3:
        """
        Compute the homogeneous transformation matrix for the camera.

        The transformation matrix is computed from the camera's position and orientation.
        The position and orientation are retrieved from the camera data.

        Returns:
        np.ndarray: The 4x4 homogeneous transformation matrix representing the camera's pose.
        """
        return get_pose(self._model, self._data, self.name, ObjType.CAMERA)

    @property
    def P(self) -> np.ndarray:
        """
        Compute the projection matrix for the camera.

        The projection matrix is computed as the product of the camera's intrinsic matrix (K)
        and the homogeneous transformation matrix (T_world_cam).

        Returns:
        np.ndarray: The 3x4 projection matrix.
        """
        return self.K @ self.T_world_cam.A[:3, :]

    @property
    def image(self) -> np.ndarray:
        """Return the captured RGB image."""
        self._renderer.update_scene(self._data, camera=self.name)
        self._image = self._renderer.render()
        return self._image

    @property
    def depth_image(self) -> np.ndarray:
        """Return the captured depth image."""
        self._renderer.update_scene(self._data, camera=self.name)
        self._renderer.enable_depth_rendering()
        self._depth_image = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return self._depth_image

    @property
    def seg_image(self) -> np.ndarray:
        """Return the captured segmentation image based on object's id."""
        self._renderer.update_scene(self._data, camera=self.name)
        self._renderer.enable_segmentation_rendering()

        self._seg_id_image = self._renderer.render()[:, :, 0].reshape(
            (self.height, self.width)
        )
        self._renderer.disable_segmentation_rendering()
        return self._seg_id_image

    @property
    def point_cloud(self) -> np.ndarray:
        """Return the captured point cloud."""
        self._point_cloud = self._depth_to_point_cloud(self.depth_image)
        return self._point_cloud

    @property
    def fov(self) -> float:
        """Get the field of view (FOV) of the camera.

        Returns:
        - float: The field of view angle in degrees.
        """
        return self._model.cam(self.id).fovy[0]

    @property
    def id(self) -> int:
        """Get the identifier of the camera.

        Returns:
        - int: The identifier of the camera.
        """
        return self._cam_id

    def _depth_to_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Method to convert depth image to a point cloud in camera coordinates.

        Args:
        - depth_image: The depth image we want to convert to a point cloud.

        Returns:
        - np.ndarray: 3D points in camera coordinates.
        """
        # Get image dimensions
        dimg_shape = depth_image.shape
        height = dimg_shape[0]
        width = dimg_shape[1]

        # Create pixel grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Flatten arrays for vectorized computation
        x_flat = x.flatten()
        y_flat = y.flatten()
        depth_flat = depth_image.flatten()

        # Negate depth values because z-axis goes into the camera
        depth_flat = -depth_flat

        # Stack flattened arrays to form homogeneous coordinates
        homogeneous_coords = np.vstack((y_flat, x_flat, np.ones_like(x_flat)))

        # Compute inverse of the intrinsic matrix K
        K_inv = np.linalg.inv(self.K)

        # Calculate 3D points in camera coordinates
        points_camera = np.dot(K_inv, homogeneous_coords) * depth_flat

        # Homogeneous coordinates to 3D points
        points_camera = np.vstack((points_camera, np.ones_like(x_flat)))

        points_camera = points_camera.T

        # dehomogenize
        points_camera = points_camera[:, :3] / points_camera[:, 3][:, np.newaxis]

        return points_camera

    def capture(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture a new RGB image, depth image, point cloud, and segmentation image from the camera.

        This method captures the current view from the camera in the simulation, returning an RGB image,
        a depth image, a point cloud, and a segmentation image. The RGB image contains color information,
        the depth image captures the distance to objects, the point cloud provides 3D spatial data, and the
        segmentation image contains object labels for each pixel.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - RGB image (np.ndarray): The captured color image.
            - Depth image (np.ndarray): The captured depth information.
            - Point cloud (np.ndarray): The 3D point cloud from the camera's perspective.
            - Segmentation image (np.ndarray): The image containing object segmentation labels.
        """
        return self.image, self.depth_image, self.point_cloud, self.seg_image

    def save(self, img_name: str = None) -> None:
        """Saves the captured image and depth information.

        Args:
        - img_name: Name for the saved image file.
        """
        print(f"saving rgb image, depth image and point cloud to {self.save_dir}")

        if img_name is None:
            timestamp = datetime.now()
            cv2.imwrite(
                self._save_dir + f"{timestamp}_rgb.png",
                cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(self._save_dir + f"{timestamp}_seg.png", self.seg_image)
            np.save(self._save_dir + f"{timestamp}_depth.npy", self.depth_image)
            pcwrite(self._save_dir + f"{timestamp}.pcd", self.point_cloud)

        else:
            cv2.imwrite(
                self._save_dir + f"{img_name}_rgb.png",
                cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(self._save_dir + f"{img_name}_seg.png", self.seg_image)
            np.save(self._save_dir + f"{img_name}_depth.npy", self.depth_image)
            pcwrite(self._save_dir + f"{img_name}.pcd", self.point_cloud)


if __name__ == "__main__":

    class MjSim(BaseSim):
        def __init__(self):
            super().__init__()

            self._model, self._data = self.init()

            self.cam = Camera(self.model, self.data)

        def init(self) -> Tuple[mj.MjModel, mj.MjData]:
            # root
            _MJ_SIM = Path(__file__).parent.parent.parent
            # scene path
            _XML_SCENE = Path(_MJ_SIM / "scenes/empty.xml")
            scene = mjcf.from_path(_XML_SCENE)

            # load in a prop
            _SHADOW_HAND = Path(_MJ_SIM / "assets" / "shadow_hand" / "right_hand.xml")
            shadow_hand = mjcf.from_path(_SHADOW_HAND)
            shadow_hand_base_body = shadow_hand.worldbody.find(
                "body", "rh_forearm_base"
            )
            shadow_hand_base_body.pos = [0, 0, 0.1]

            scene.attach(shadow_hand)

            scene.worldbody.add(
                "camera",
                name="cam",
                mode="targetbody",
                target="right_shadow_hand/rh_palm",
                pos="0.5 0.5 0.5",
            )

            m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
            d = mj.MjData(m)

            # step once to compute the poses of objects
            mj.mj_step(m, d)

            return m, d

        @property
        def data(self):
            return self._data

        @property
        def model(self):
            return self._model

        def keyboard_callback(self, key):
            if key is glfw.KEY_SPACE:
                print(f"camera '{self.cam.name}' just saved camera data...")
                self.cam.save("demo")

    sim = MjSim()

    sim.run()
