import mujoco as mj
import numpy as np
import pytest

from robots import URRobot

_TEST_SCENE = r"""
    <mujoco model="empty scene">
    <compiler angle="radian"/>

    <option gravity="0 0 -9.82" integrator="implicitfast" cone="elliptic"/>

    <size nkey="1"/>

    <visual>
        <global azimuth="120" elevation="-20"/>
        <headlight diffuse="0.6 0.6 0.6" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
    </visual>

    <statistic meansize="0.08" extent="0.8" center="0.3 0 0.3"/>

    <default>
        <default class="/"/>
        <default class="ur5e/">
        <default class="ur5e/ur5e">
            <material shininess="0.25"/>
            <joint range="-6.28319 6.28319" armature="0.1"/>
            <site size="0.001 0.005 0.005" group="4" rgba="0.5 0.5 0.5 0.3"/>
            <general ctrlrange="-6.2831 6.2831" forcerange="-150 150" biastype="affine" gainprm="2000" biasprm="0 -2000 -400"/>
            <default class="ur5e/size3">
            <default class="ur5e/size3_limited">
                <joint range="-3.1415 3.1415"/>
                <general ctrlrange="-3.1415 3.1415"/>
            </default>
            </default>
            <default class="ur5e/size1">
            <general forcerange="-28 28" gainprm="500" biasprm="0 -500 -100"/>
            </default>
            <default class="ur5e/visual">
            <geom type="mesh" contype="0" conaffinity="0" group="2"/>
            </default>
            <default class="ur5e/collision">
            <geom type="capsule" group="3"/>
            <default class="ur5e/eef_collision">
                <geom type="cylinder"/>
            </default>
            </default>
        </default>
        </default>
        <default class="ur5e/unnamed_model/"/>
        <default class="unnamed_model/"/>
        <default class="unnamed_model_1/"/>
    </default>

    <asset>
        <texture type="skybox" name="//unnamed_texture_0" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" class="/" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        <material name="ur5e/black" class="ur5e/ur5e" rgba="0.033 0.033 0.033 1"/>
        <material name="ur5e/jointgray" class="ur5e/ur5e" rgba="0.278 0.278 0.278 1"/>
        <material name="ur5e/linkgray" class="ur5e/ur5e" rgba="0.82 0.82 0.82 1"/>
        <material name="ur5e/urblue" class="ur5e/ur5e" rgba="0.49 0.678 0.8 1"/>
        assets/universal_robots_ur5e/assets/
        <mesh name="ur5e/base_0" class="ur5e/" file="assets/universal_robots_ur5e/assets/base_0.obj"/>
        <mesh name="ur5e/base_1" class="ur5e/" file="assets/universal_robots_ur5e/assets/base_1.obj"/>
        <mesh name="ur5e/shoulder_0" class="ur5e/" file="assets/universal_robots_ur5e/assets/shoulder_0.obj"/>
        <mesh name="ur5e/shoulder_1" class="ur5e/" file="assets/universal_robots_ur5e/assets/shoulder_1.obj"/>
        <mesh name="ur5e/shoulder_2" class="ur5e/" file="assets/universal_robots_ur5e/assets/shoulder_2.obj"/>
        <mesh name="ur5e/upperarm_0" class="ur5e/" file="assets/universal_robots_ur5e/assets/upperarm_0.obj"/>
        <mesh name="ur5e/upperarm_1" class="ur5e/" file="assets/universal_robots_ur5e/assets/upperarm_1.obj"/>
        <mesh name="ur5e/upperarm_2" class="ur5e/" file="assets/universal_robots_ur5e/assets/upperarm_2.obj"/>
        <mesh name="ur5e/upperarm_3" class="ur5e/" file="assets/universal_robots_ur5e/assets/upperarm_3.obj"/>
        <mesh name="ur5e/forearm_0" class="ur5e/" file="assets/universal_robots_ur5e/assets/forearm_0.obj"/>
        <mesh name="ur5e/forearm_1" class="ur5e/" file="assets/universal_robots_ur5e/assets/forearm_1.obj"/>
        <mesh name="ur5e/forearm_2" class="ur5e/" file="assets/universal_robots_ur5e/assets/forearm_2.obj"/>
        <mesh name="ur5e/forearm_3" class="ur5e/" file="assets/universal_robots_ur5e/assets/forearm_3.obj"/>
        <mesh name="ur5e/wrist1_0" class="ur5e/" file="assets/universal_robots_ur5e/assets/wrist1_0.obj"/>
        <mesh name="ur5e/wrist1_1" class="ur5e/" file="assets/universal_robots_ur5e/assets/wrist1_1.obj"/>
        <mesh name="ur5e/wrist1_2" class="ur5e/" file="assets/universal_robots_ur5e/assets/wrist1_2.obj"/>
        <mesh name="ur5e/wrist2_0" class="ur5e/" file="assets/universal_robots_ur5e/assets/wrist2_0.obj"/>
        <mesh name="ur5e/wrist2_1" class="ur5e/" file="assets/universal_robots_ur5e/assets/wrist2_1.obj"/>
        <mesh name="ur5e/wrist2_2" class="ur5e/" file="assets/universal_robots_ur5e/assets/wrist2_2.obj"/>
        <mesh name="ur5e/wrist3" class="ur5e/" file="assets/universal_robots_ur5e/assets/wrist3.obj"/>
    </asset>

    <worldbody>
        <geom name="floor" class="/" size="0 0 0.5" type="plane" material="groundplane"/>
        <camera name="cam" class="/" pos="1 1 1"/>
        <light name="//unnamed_light_0" class="/" pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <body name="ur5e/">
        <light name="ur5e/spotlight" class="ur5e/" target="ur5e/wrist_2_link" pos="0 -1 2" dir="0 0 -1" mode="targetbodycom"/>
        <body name="ur5e/base" childclass="ur5e/ur5e" pos="0.223319 0.375375 0.0879133" quat="-0.198585 -0.00311175 0.00122999 0.980078">
            <inertial pos="0 0 0" mass="4" diaginertia="0.00443333 0.00443333 0.0072"/>
            <geom name="ur5e//unnamed_geom_0" class="ur5e/visual" material="ur5e/black" mesh="ur5e/base_0"/>
            <geom name="ur5e//unnamed_geom_1" class="ur5e/visual" material="ur5e/jointgray" mesh="ur5e/base_1"/>
            <body name="ur5e/shoulder_link" pos="0 0 0.163">
            <inertial pos="0 0 0" mass="3.7" diaginertia="0.0102675 0.0102675 0.00666"/>
            <joint name="ur5e/shoulder_pan_joint" class="ur5e/size3" pos="0 0 0" axis="0 0 1"/>
            <geom name="ur5e//unnamed_geom_2" class="ur5e/visual" material="ur5e/urblue" mesh="ur5e/shoulder_0"/>
            <geom name="ur5e//unnamed_geom_3" class="ur5e/visual" material="ur5e/black" mesh="ur5e/shoulder_1"/>
            <geom name="ur5e//unnamed_geom_4" class="ur5e/visual" material="ur5e/jointgray" mesh="ur5e/shoulder_2"/>
            <geom name="ur5e//unnamed_geom_5" class="ur5e/collision" size="0.06 0.06" pos="0 0 -0.04"/>
            <body name="ur5e/upper_arm_link" pos="0 0.138 0" quat="0.707107 0 0.707107 0">
                <inertial pos="0 0 0.2125" mass="8.393" diaginertia="0.133886 0.133886 0.0151074"/>
                <joint name="ur5e/shoulder_lift_joint" class="ur5e/size3" pos="0 0 0" axis="0 1 0"/>
                <geom name="ur5e//unnamed_geom_6" class="ur5e/visual" material="ur5e/linkgray" mesh="ur5e/upperarm_0"/>
                <geom name="ur5e//unnamed_geom_7" class="ur5e/visual" material="ur5e/black" mesh="ur5e/upperarm_1"/>
                <geom name="ur5e//unnamed_geom_8" class="ur5e/visual" material="ur5e/jointgray" mesh="ur5e/upperarm_2"/>
                <geom name="ur5e//unnamed_geom_9" class="ur5e/visual" material="ur5e/urblue" mesh="ur5e/upperarm_3"/>
                <geom name="ur5e//unnamed_geom_10" class="ur5e/collision" size="0.06 0.06" pos="0 -0.04 0" quat="0.707107 0.707107 0 0"/>
                <geom name="ur5e//unnamed_geom_11" class="ur5e/collision" size="0.05 0.2" pos="0 0 0.2"/>
                <body name="ur5e/forearm_link" pos="0 -0.131 0.425">
                <inertial pos="0 0 0.196" mass="2.275" diaginertia="0.0311796 0.0311796 0.004095"/>
                <joint name="ur5e/elbow_joint" class="ur5e/size3_limited" pos="0 0 0" axis="0 1 0"/>
                <geom name="ur5e//unnamed_geom_12" class="ur5e/visual" material="ur5e/urblue" mesh="ur5e/forearm_0"/>
                <geom name="ur5e//unnamed_geom_13" class="ur5e/visual" material="ur5e/linkgray" mesh="ur5e/forearm_1"/>
                <geom name="ur5e//unnamed_geom_14" class="ur5e/visual" material="ur5e/black" mesh="ur5e/forearm_2"/>
                <geom name="ur5e//unnamed_geom_15" class="ur5e/visual" material="ur5e/jointgray" mesh="ur5e/forearm_3"/>
                <geom name="ur5e//unnamed_geom_16" class="ur5e/collision" size="0.055 0.06" pos="0 0.08 0" quat="0.707107 0.707107 0 0"/>
                <geom name="ur5e//unnamed_geom_17" class="ur5e/collision" size="0.038 0.19" pos="0 0 0.2"/>
                <body name="ur5e/wrist_1_link" pos="0 0 0.392" quat="0.707107 0 0.707107 0">
                    <inertial pos="0 0.127 0" mass="1.219" diaginertia="0.0025599 0.0025599 0.0021942"/>
                    <joint name="ur5e/wrist_1_joint" class="ur5e/size1" pos="0 0 0" axis="0 1 0"/>
                    <geom name="ur5e//unnamed_geom_18" class="ur5e/visual" material="ur5e/black" mesh="ur5e/wrist1_0"/>
                    <geom name="ur5e//unnamed_geom_19" class="ur5e/visual" material="ur5e/urblue" mesh="ur5e/wrist1_1"/>
                    <geom name="ur5e//unnamed_geom_20" class="ur5e/visual" material="ur5e/jointgray" mesh="ur5e/wrist1_2"/>
                    <geom name="ur5e//unnamed_geom_21" class="ur5e/collision" size="0.04 0.07" pos="0 0.05 0" quat="0.707107 0.707107 0 0"/>
                    <body name="ur5e/wrist_2_link" pos="0 0.127 0">
                    <inertial pos="0 0 0.1" mass="1.219" diaginertia="0.0025599 0.0025599 0.0021942"/>
                    <joint name="ur5e/wrist_2_joint" class="ur5e/size1" pos="0 0 0" axis="0 0 1"/>
                    <geom name="ur5e//unnamed_geom_22" class="ur5e/visual" material="ur5e/black" mesh="ur5e/wrist2_0"/>
                    <geom name="ur5e//unnamed_geom_23" class="ur5e/visual" material="ur5e/urblue" mesh="ur5e/wrist2_1"/>
                    <geom name="ur5e//unnamed_geom_24" class="ur5e/visual" material="ur5e/jointgray" mesh="ur5e/wrist2_2"/>
                    <geom name="ur5e//unnamed_geom_25" class="ur5e/collision" size="0.04 0.06" pos="0 0 0.04"/>
                    <geom name="ur5e//unnamed_geom_26" class="ur5e/collision" size="0.04 0.04" pos="0 0.02 0.1" quat="0.707107 0.707107 0 0"/>
                    <body name="ur5e/wrist_3_link" pos="0 0 0.1">
                        <inertial pos="0 0.0771683 0" quat="0.707107 0 0 0.707107" mass="0.1889" diaginertia="0.000132134 9.90863e-05 9.90863e-05"/>
                        <joint name="ur5e/wrist_3_joint" class="ur5e/size1" pos="0 0 0" axis="0 1 0"/>
                        <geom name="ur5e//unnamed_geom_27" class="ur5e/visual" material="ur5e/linkgray" mesh="ur5e/wrist3"/>
                        <geom name="ur5e//unnamed_geom_28" class="ur5e/eef_collision" size="0.04 0.02" pos="0 0.08 0" quat="0.707107 0.707107 0 0"/>
                        <site name="ur5e/attachment_site" pos="0 0.1 0" quat="-0.707107 0.707107 0 0"/>
                        <body name="ur5e/unnamed_model/" pos="0 0.1 0" quat="-0.707107 0.707107 0 0">
                        <body name="ur5e/unnamed_model/flange_tool">
                            <geom name="ur5e/unnamed_model/flange_tool_base" class="ur5e/unnamed_model/" size="0.032 0.01" type="cylinder" mass="0.5" rgba="0.2 0.2 0.2 1"/>
                            <geom name="ur5e/unnamed_model/flange_tool_tip" class="ur5e/unnamed_model/" size="0.015 0.05" pos="0 0 0.05" type="cylinder" mass="0.5" rgba="0.2 0.2 0.2 1"/>
                            <site name="ur5e/unnamed_model/flange_tool" class="ur5e/unnamed_model/" pos="0 0 -0.01"/>
                        </body>
                        </body>
                    </body>
                    </body>
                </body>
                </body>
            </body>
            </body>
        </body>
        </body>
        <body name="unnamed_model/">
        <body name="unnamed_model/flexcell_top">
            <geom name="unnamed_model/flexcell_top" class="unnamed_model/" size="0.6 0.4 0.025" pos="0.6 0.4 0.025" type="box" solref="0.0001" solimp="0.9 0.95 0.001 0.5 4" margin="0.001"/>
        </body>
        </body>
        <body name="unnamed_model_1/">
        <body name="unnamed_model_1/ur_mounting_plate" pos="0.223319 0.375375 0.0689566">
            <geom name="unnamed_model_1/ur_mounting_plate" class="unnamed_model_1/" size="0.1 0.0189566" type="cylinder" mass="10" rgba="0.2 0.2 0.2 1"/>
        </body>
        </body>
    </worldbody>

    <actuator>
        <general name="ur5e/shoulder_pan" class="ur5e/size3" joint="ur5e/shoulder_pan_joint"/>
        <general name="ur5e/shoulder_lift" class="ur5e/size3" joint="ur5e/shoulder_lift_joint"/>
        <general name="ur5e/elbow" class="ur5e/size3_limited" joint="ur5e/elbow_joint"/>
        <general name="ur5e/wrist_1" class="ur5e/size1" joint="ur5e/wrist_1_joint"/>
        <general name="ur5e/wrist_2" class="ur5e/size1" joint="ur5e/wrist_2_joint"/>
        <general name="ur5e/wrist_3" class="ur5e/size1" joint="ur5e/wrist_3_joint"/>
    </actuator>

    <sensor>
        <force site="ur5e/attachment_site" name="ur5e/force"/>
        <torque site="ur5e/attachment_site" name="ur5e/torque"/>
    </sensor>

    <keyframe>
        <key name="ur5e/home" qpos="-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0" ctrl="-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"/>
    </keyframe>
    </mujoco>

"""


@pytest.fixture
def setup_scene():
    m = mj.MjModel.from_xml_string(_TEST_SCENE)
    d = mj.MjData(m)
    mj.mj_step(m, d)
    return m, d


def test_robot_q(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert len(robot.q) == 6


def test_robot_dq(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert len(robot.dq) == 6


def test_robot_ddq(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert len(robot.ddq) == 6


def test_robot_ctrl(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert len(robot.q) == 6


def test_robot_c(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert len(robot.c) == 6


def test_robot_Mq(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert robot.Mq.shape == np.zeros(shape=(6, 6)).shape


def test_robot_Mx(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert robot.Mx.shape == np.zeros(shape=(6, 6)).shape


def test_robot_Jp(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert robot.Jp.shape == np.zeros(shape=(3, 6)).shape


def test_robot_Jo(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert robot.Jo.shape == np.zeros(shape=(3, 6)).shape


def test_robot_J(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    assert robot.J.shape == np.zeros(shape=(6, 6)).shape


def test_robot_set_ctrl(setup_scene):
    m, d = setup_scene
    robot = URRobot(m, d)
    robot.set_ctrl(np.ones(robot.info.n_actuators))
    assert np.allclose(
        [d.actuator(aid).ctrl for aid in robot.info.actuator_ids],
        np.ones(robot.info.n_actuators),
    )
