import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union
from xml.dom import minidom

import mujoco as mj
import numpy as np
import spatialmath as sm
import spatialmath.base as smb

from utils.sm import make_tf


class JointType(Enum):
    """
    Enumeration of joint types used in MuJoCo simulations.

    This class defines the types of joints that can be used in a MuJoCo simulation,
    including free, ball, slide, and hinge joints. These joint types are associated
    with integer values that are used within the MuJoCo framework to specify the type
    of joint for different elements of the model.

    Attributes
    ----------
    FREE : int
        A joint that allows for unrestricted movement in all translational and rotational
        degrees of freedom. Represented by the integer value 0.
    BALL : int
        A ball-and-socket joint that allows rotation in all directions, but no translational
        movement. Represented by the integer value 1.
    SLIDE : int
        A prismatic joint that allows translational movement along a single axis, with no
        rotational freedom. Represented by the integer value 2.
    HINGE : int
        A rotational joint that allows rotation around a single axis, similar to a door hinge.
        Represented by the integer value 3.
    """

    FREE = int(mj.mjtJoint.mjJNT_FREE)
    BALL = int(mj.mjtJoint.mjJNT_BALL)
    SLIDE = int(mj.mjtJoint.mjJNT_SLIDE)
    HINGE = int(mj.mjtJoint.mjJNT_HINGE)


class ObjType(Enum):
    """
    Enumeration of object types used in MuJoCo simulations.

    This class defines various object types that can be used in a MuJoCo simulation,
    including bodies, joints, geoms, sites, actuators, and sensors. These object types
    are associated with integer values that are used within the MuJoCo framework to
    specify the type of object for different elements of the model.

    Attributes
    ----------
    UNKNOWN : int
        Represents an unknown object type. Used when the object type is not recognized.
        Represented by the integer value 0.
    BODY : int
        Represents a body in the simulation. Bodies are rigid objects that can be connected
        by joints or other constraints. Represented by the integer value 1.
    XBODY : int
        Represents a body but used to access the regular frame instead of the i-frame.
        Represented by the integer value 2.
    JOINT : int
        Represents a joint that connects two bodies and allows for relative movement.
        Represented by the integer value 3.
    DOF : int
        Represents a degree of freedom (DOF) of a joint, typically used to describe the
        axes along which a joint can move or rotate. Represented by the integer value 4.
    GEOM : int
        Represents a geometric shape associated with a body, used for collision detection
        and visual rendering. Represented by the integer value 5.
    SITE : int
        Represents a site, which is a fixed point or frame on a body used for specifying
        positions or attaching sensors. Represented by the integer value 6.
    CAMERA : int
        Represents a camera, which can be used to render views of the simulation environment.
        Represented by the integer value 7.
    LIGHT : int
        Represents a light source in the simulation environment. Represented by the integer
        value 8.
    FLEX : int
        Represents a flex object, typically used in simulations of flexible materials.
        Represented by the integer value 9.
    MESH : int
        Represents a mesh object, typically a more complex geometric shape defined by vertices
        and faces. Represented by the integer value 10.
    SKIN : int
        Represents a skin, usually used for rendering soft or flexible outer surfaces on bodies.
        Represented by the integer value 11.
    HFIELD : int
        Represents a heightfield, typically used to model uneven terrain or surfaces in the
        simulation. Represented by the integer value 12.
    TEXTURE : int
        Represents a texture, which can be applied to geometric shapes for visual realism.
        Represented by the integer value 13.
    MATERIAL : int
        Represents a material for rendering, which defines visual properties like color and
        reflectiveness. Represented by the integer value 14.
    PAIR : int
        Represents a geom pair to include in the simulation for collision detection.
        Represented by the integer value 15.
    EXCLUDE : int
        Represents a body pair to exclude from collision detection. Represented by the integer
        value 16.
    EQUALITY : int
        Represents an equality constraint, used to enforce specific relationships between
        different elements in the simulation. Represented by the integer value 17.
    TENDON : int
        Represents a tendon, typically used to model soft structures or actuators in biological
        systems. Represented by the integer value 18.
    ACTUATOR : int
        Represents an actuator, which generates forces or torques to move bodies in the simulation.
        Represented by the integer value 19.
    SENSOR : int
        Represents a sensor, which is used to measure various aspects of the simulation such as
        positions, velocities, or forces. Represented by the integer value 20.
    NUMERIC : int
        Represents a numeric element, often used for user-defined values or parameters.
        Represented by the integer value 21.
    TEXT : int
        Represents a text element, used to store or display text in the simulation.
        Represented by the integer value 22.
    TUPLE : int
        Represents a tuple, often used to group multiple elements together.
        Represented by the integer value 23.
    KEY : int
        Represents a keyframe, used to save or restore the state of the simulation at specific
        points in time. Represented by the integer value 24.
    PLUGIN : int
        Represents a plugin instance, which can be used to extend the functionality of the
        simulation with user-defined code. Represented by the integer value 25.
    """

    UNKNOWN = int(mj.mjtObj.mjOBJ_UNKNOWN)  # unknown object type
    BODY = int(mj.mjtObj.mjOBJ_BODY)  # body
    XBODY = int(
        mj.mjtObj.mjOBJ_XBODY
    )  # body, used to access regular frame instead of i-frame
    JOINT = int(mj.mjtObj.mjOBJ_JOINT)  # joint
    DOF = int(mj.mjtObj.mjOBJ_DOF)  # dof
    GEOM = int(mj.mjtObj.mjOBJ_GEOM)  # geom
    SITE = int(mj.mjtObj.mjOBJ_SITE)  # site
    CAMERA = int(mj.mjtObj.mjOBJ_CAMERA)  # camera
    LIGHT = int(mj.mjtObj.mjOBJ_LIGHT)  # light
    FLEX = int(mj.mjtObj.mjOBJ_FLEX)  # flex
    MESH = int(mj.mjtObj.mjOBJ_MESH)  # mesh
    SKIN = int(mj.mjtObj.mjOBJ_SKIN)  # skin
    HFIELD = int(mj.mjtObj.mjOBJ_HFIELD)  # heightfield
    TEXTURE = int(mj.mjtObj.mjOBJ_TEXTURE)  # texture
    MATERIAL = int(mj.mjtObj.mjOBJ_MATERIAL)  # material for rendering
    PAIR = int(mj.mjtObj.mjOBJ_PAIR)  # geom pair to include
    EXCLUDE = int(mj.mjtObj.mjOBJ_EXCLUDE)  # body pair to exclude
    EQUALITY = int(mj.mjtObj.mjOBJ_EQUALITY)  # equality constraint
    TENDON = int(mj.mjtObj.mjOBJ_TENDON)  # tendon
    ACTUATOR = int(mj.mjtObj.mjOBJ_ACTUATOR)  # actuator
    SENSOR = int(mj.mjtObj.mjOBJ_SENSOR)  # sensor
    NUMERIC = int(mj.mjtObj.mjOBJ_NUMERIC)  # numeric
    TEXT = int(mj.mjtObj.mjOBJ_TEXT)  # text
    TUPLE = int(mj.mjtObj.mjOBJ_TUPLE)  # tuple
    KEY = int(mj.mjtObj.mjOBJ_KEY)  # keyframe
    PLUGIN = int(mj.mjtObj.mjOBJ_PLUGIN)  # plugin instance
    NOOBJECT = int(mj.mjtObj.mjNOBJECT)  # number of object types
    FRAME = int(mj.mjtObj.mjOBJ_FRAME)  # frame


def get_number_of(model: mj.MjModel, obj_type: ObjType) -> int:
    """
    Retrieves the count of objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model from which to count objects.
    obj_type : ObjType
        The type of objects to count, e.g., actuators, bodies, joints.

    Returns
    -------
    int
        The number of objects of the specified type in the model.

    Raises
    ------
    ValueError
        If the specified object type is not recognized.
    """
    type_to_attribute = {
        ObjType.ACTUATOR: model.nu,
        ObjType.BODY: model.nbody,
        ObjType.JOINT: model.njnt,
        ObjType.DOF: model.nv,
        ObjType.GEOM: model.ngeom,
        ObjType.SITE: model.nsite,
        ObjType.CAMERA: model.ncam,
        ObjType.LIGHT: model.nlight,
        ObjType.MESH: model.nmesh,
        ObjType.SKIN: model.nskin,
        ObjType.HFIELD: model.nhfield,
        ObjType.TEXTURE: model.ntex,
        ObjType.MATERIAL: model.nmat,
        ObjType.PAIR: model.npair,
        ObjType.EXCLUDE: model.nexclude,
        ObjType.EQUALITY: model.neq,
        ObjType.TENDON: model.ntendon,
        ObjType.SENSOR: model.nsensordata,
        ObjType.NUMERIC: model.nnumeric,
        ObjType.TEXT: model.ntext,
        ObjType.TUPLE: model.ntuple,
        ObjType.KEY: model.nkey,
        ObjType.PLUGIN: model.nplugin,
        ObjType.FRAME: 0,  # No direct model attribute
    }

    if obj_type not in type_to_attribute:
        raise ValueError(f"Object type {obj_type} not recognized.")

    return type_to_attribute[obj_type]


def get_names(model: mj.MjModel, obj_type: ObjType) -> List[str]:
    """
    Retrieves the names of all objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the objects.
    obj_type : ObjType
        The type of objects to retrieve names for, e.g., actuators, bodies.

    Returns
    -------
    List[str]
        A list of names for all objects of the specified type in the model.
    """
    return [
        mj.mj_id2name(model, obj_type.value, id)
        for id in range(get_number_of(model, obj_type))
    ]


def get_ids(model: mj.MjModel, obj_type: ObjType) -> List[int]:
    """
    Retrieves the IDs of all objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the objects.
    obj_type : ObjType
        The type of objects to retrieve IDs for.

    Returns
    -------
    List[int]
        A list of IDs for all objects of the specified type in the model.
    """
    return [i for i in range(get_number_of(model, obj_type))]


def name2id(model: mj.MjModel, name: str, obj_type: ObjType) -> int:
    """
    Retrieves the ID of an object by its name and type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the object.
    name : str
        The name of the object.
    obj_type : ObjType
        The type of the object, e.g., body, joint.

    Returns
    -------
    int
        The ID of the object if it exists.
    """
    id = mj.mj_name2id(model, obj_type.value, name)
    assert does_exist(model, name, obj_type)
    return id


def id2name(model: mj.MjModel, id: int, obj_type: ObjType) -> str:
    """
    Retrieves the name of an object by its ID and type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the object.
    id : int
        The ID of the object.
    obj_type : ObjType
        The type of the object, e.g., actuator, site.

    Returns
    -------
    str
        The name of the object if it exists.
    """
    name = mj.mj_id2name(model, obj_type.value, id)
    assert does_exist(model, id, obj_type)
    return name


def does_exist(
    model: mj.MjModel, identifier: Union[int, str], obj_type: ObjType
) -> bool:
    """
    Checks if an object with a given name or ID exists in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model to check within.
    identifier : int or str
        The ID or name of the object.
    obj_type : ObjType
        The type of the object, e.g., geom, sensor.

    Returns
    -------
    bool
        True if the object exists, False otherwise.

    Raises
    ------
    ValueError
        If the object type is unrecognized or if the specified identifier type is invalid.
    """
    if isinstance(identifier, str):
        exists = (
            True if mj.mj_name2id(model, obj_type.value, identifier) != -1 else False
        )
        if not exists:
            raise ValueError(
                f"{obj_type.name} name '{identifier}' not found in the model. The model contain the {obj_type.name}s {get_names(model, obj_type)}"
            )
    elif isinstance(identifier, int):
        exists = (
            True
            if mj.mj_id2name(model, obj_type.value, identifier) is not None
            else False
        )
        if not exists:
            raise ValueError(
                f"{obj_type.name} id '{identifier}' not found in the model. The model contain the {obj_type.name}s {get_ids(model, obj_type)}"
            )
    else:
        raise ValueError(
            f"Invalid type input id with value '{identifier}' and type {type(identifier)} use either string or int."
        )
    return exists


def get_pose(
    model: mj.MjModel, data: mj.MjData, identifier: Union[int, str], obj_type: ObjType
) -> sm.SE3:
    """
    Retrieves the pose (position and orientation) of an object in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the object.
    data : mj.MjData
        The simulation data containing the pose information.
    identifier : int or str
        The ID or name of the object.
    obj_type : ObjType
        The type of the object, e.g., body, site, camera.

    Returns
    -------
    sm.SE3
        The pose of the object as an SE3 transformation matrix.

    Raises
    ------
    ValueError
        If the object type does not support pose retrieval.
    """

    if not does_exist(model, identifier, obj_type):
        return sm.SE3()

    # Define a mapping for types that can provide poses
    pose_mapping = {
        ObjType.ACTUATOR: data.actuator,
        ObjType.BODY: data.body,
        ObjType.JOINT: data.joint,
        ObjType.GEOM: data.geom,
        ObjType.SITE: data.site,
        ObjType.CAMERA: data.cam,
        ObjType.LIGHT: data.light,
    }

    # Check if type has a pose; raise an error if it does not
    if obj_type not in pose_mapping:
        raise ValueError(f"obj_type {obj_type.name} cannot provide a pose...")

    if (
        (obj_type is ObjType.CAMERA)
        or (obj_type is ObjType.SITE)
        or (obj_type is ObjType.GEOM)
    ):
        obj = pose_mapping[obj_type](identifier)
        t, q = obj.xpos, smb.r2q(np.reshape(obj.xmat, (3, 3)))
        return make_tf(pos=t, ori=q)

    # Retrieve position and quaternion
    obj = pose_mapping[obj_type](identifier)
    t, q = obj.xpos, obj.xquat
    return make_tf(pos=t, ori=q)


def set_pose(
    model: mj.MjModel,
    data: mj.MjData,
    identifier: Union[int, str],
    obj_type: ObjType,
    T: sm.SE3,
) -> None:
    """
    Sets the pose (position and orientation) of an object in a MuJoCo model, if allowed.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the object.
    data : mj.MjData
        The simulation data where the pose is set.
    identifier : int or str
        The ID or name of the object.
    obj_type : ObjType
        The type of the object, e.g., body, joint.
    T : sm.SE3
        The desired pose as an SE3 transformation matrix.
    """

    assert does_exist(model, identifier, obj_type)

    def set_position_and_orientation(pos, quat):
        """Helper to set position and orientation."""
        pos[:] = T.t
        quat[:] = smb.r2q(T.R)

    # Process based on object type
    if obj_type is ObjType.BODY:
        body_id = (
            identifier
            if isinstance(identifier, int)
            else mj.mj_name2id(model, ObjType.BODY.value, identifier)
        )

        # Check if the body is a mocap body
        mocap_id = model.body_mocapid[body_id]
        if mocap_id != -1:
            set_position_and_orientation(
                data.mocap_pos[mocap_id], data.mocap_quat[mocap_id]
            )
            return

        # Check if the body has a freejoint
        if JointType(model.jnt_type[model.body_jntadr[body_id]]) is JointType.FREE:
            jnt_id = model.body_jntadr[body_id]
            set_position_and_orientation(
                data.joint(jnt_id).qpos[:3], data.joint(jnt_id).qpos[3:]
            )
            return

    elif obj_type is ObjType.JOINT:
        joint_id = (
            identifier
            if isinstance(identifier, int)
            else mj.mj_name2id(model, ObjType.JOINT.value, identifier)
        )

        # Check if the joint is a freejoint
        if JointType(model.joint(joint_id).type[0]) is JointType.FREE:
            set_position_and_orientation(
                data.joint(joint_id).qpos[:3], data.joint(joint_id).qpos[3:]
            )
            return

    # If no valid option found, raise an error
    raise ValueError(
        f"{obj_type.name} '{identifier}' cannot have its pose set. Only mocap bodies, bodies with freejoints, or freejoints are allowed."
    )


def get_joint_qpos_addr(model: mj.MjModel, joint_name: Union[int, str]) -> int:
    """
    Retrieves the address of the generalized position (qpos) for a specific joint in the MuJoCo model.

    This function returns the index of the joint's generalized position in the `qpos` array
    of the MuJoCo model. This index can be used to access or modify the joint's position
    directly in the simulation data.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object from which to retrieve the joint's qpos address.
    joint_name : Union[int, str]
        The name or ID of the joint whose qpos address is being retrieved. Can be specified
        either as a string (name) or an integer (ID).

    Returns
    ----------
    int
        The address (index) of the joint's generalized position in the `qpos` array.

    Notes
    -----
    Ensure that the specified `joint_name` corresponds to a valid joint in the MuJoCo model.
    """
    joint_id = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )

    return model.jnt_qposadr[joint_id]


def set_joint_q(
    data: mj.MjData,
    model: mj.MjModel,
    joint_name: Union[int, str],
    q: Union[np.ndarray, float],
    unit: str = "rad",
) -> None:
    """
    Sets the position(s) (angle(s)) of a joint in the MuJoCo simulation.

    This function updates the generalized position (`qpos`) of a specified joint in
    the MuJoCo simulation. The joint's position can be set in radians or degrees,
    depending on the `unit` parameter. If degrees are provided, they will be converted
    to radians before being applied. The positions are assigned to the joint's indices
    in the `qpos` array of the simulation data.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose position is being set.
    q : Union[np.ndarray, float]
        The position(s) (angle(s)) to be set for the joint. Can be a single value or an array
        of values, depending on the type of joint.
    unit : str, optional
        The unit of the position value, either "rad" (radians) or "deg" (degrees). Defaults to "rad".

    Returns
    ----------
    None

    Raises
    ------
    ValueError
        If the dimensions of `q` do not match the number of positions for the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - If the unit is "deg", the function will convert the angles from degrees to radians before applying.
    - Ensure that the `q` values match the dimensions expected for the joint.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )

    # Convert q to radians if the unit is degrees
    if unit == "deg":
        q = np.deg2rad(q)

    q_indxs = get_joint_qpos_indxs(data, model, joint_name)

    # Ensure q is a numpy array
    if isinstance(q, (int, float)):
        q = np.array([q])
    if isinstance(q, list):
        q = np.array(q)

    # Validate the dimensions of q
    if q.shape[0] != len(q_indxs):
        raise ValueError(
            f"Dimension mismatch: Expected dimension {len(q_indxs)}, "
            f"but got {q.shape[0]} for joint '{joint_name}'."
        )

    data.qpos[q_indxs] = q


def set_joint_dq(
    data: mj.MjData,
    model: mj.MjModel,
    joint_name: Union[int, str],
    dq: Union[np.ndarray, float],
    unit: str = "rad",
) -> None:
    """
    Sets the velocity of a joint in the MuJoCo simulation.

    This function updates the generalized velocity (`qvel`) of a specified joint
    in the MuJoCo simulation. The velocity can be specified in radians or degrees,
    and will be converted to radians if necessary. The velocities are assigned to the
    joint's degrees of freedom in the `qvel` array of the simulation data.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose velocity is being set.
    dq : Union[np.ndarray, float]
        The velocity value(s) to be set for the joint. This can be a single value or
        an array of values, depending on the type of joint.
    unit : str, optional
        The unit of the velocity value, either "rad" (radians) or "deg" (degrees). Defaults to "rad".

    Returns
    ----------
    None

    Raises
    ------
    ValueError
        If the dimensions of `dq` do not match the number of degrees of freedom for the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - If the unit is "deg", the function will convert the velocities from degrees to radians before applying.
    - Ensure that the `dq` values match the dimensions expected for the joint.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )

    # Convert dq to radians if the unit is degrees
    if unit == "deg":
        dq = np.deg2rad(dq)

    dq_indxs = get_joint_dof_indxs(model, joint_name)

    # Ensure dq is a numpy array
    if isinstance(dq, (int, float)):
        dq = np.array([dq])
    if isinstance(dq, list):
        dq = np.array(dq)

    # Validate the dimensions of dq
    if dq.shape[0] != len(dq_indxs):
        raise ValueError(
            f"Dimension mismatch: Expected dimension {len(dq_indxs)}, "
            f"but got {dq.shape[0]} for joint '{joint_name}'."
        )
    data.qvel[dq_indxs] = dq


def set_joint_ddq(
    data: mj.MjData,
    model: mj.MjModel,
    joint_name: Union[int, str],
    ddq: Union[np.ndarray, float],
    unit: str = "rad",
) -> None:
    """
    Sets the acceleration of a joint in the MuJoCo simulation.

    This function updates the generalized acceleration (`qacc`) of a specified joint
    in the MuJoCo simulation. The acceleration can be specified in radians or degrees,
    and will be converted to radians if necessary. The accelerations are assigned to the
    joint's degrees of freedom in the `qacc` array of the simulation data.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose acceleration is being set.
    ddq : Union[np.ndarray, float]
        The acceleration value(s) to be set for the joint. This can be a single value or
        an array of values, depending on the type of joint.
    unit : str, optional
        The unit of the acceleration value, either "rad" (radians) or "deg" (degrees). Defaults to "rad".

    Returns
    ----------
    None

    Raises
    ------
    ValueError
        If the dimensions of `ddq` do not match the number of degrees of freedom for the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - If the unit is "deg", the function will convert the accelerations from degrees to radians before applying.
    - Ensure that the `ddq` values match the dimensions expected for the joint.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )

    # Convert ddq to radians if the unit is degrees
    if unit == "deg":
        ddq = np.deg2rad(ddq)

    ddq_indxs = get_joint_dof_indxs(model, joint_name)

    # Ensure ddq is a numpy array
    if isinstance(ddq, (int, float)):
        ddq = np.array([ddq])
    if isinstance(ddq, list):
        ddq = np.array(ddq)

    # Validate the dimensions of ddq
    if ddq.shape[0] != len(ddq_indxs):
        raise ValueError(
            f"Dimension mismatch: Expected dimension {len(ddq_indxs)}, "
            f"but got {ddq.shape[0]} for joint '{joint_name}'."
        )
    data.qacc[ddq_indxs] = ddq


def get_joint_q(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[int, str]
) -> np.ndarray:
    """
    Retrieves the position (angle) of a joint in the MuJoCo simulation.

    This function extracts the current position (or angle) of a specified joint
    from the MuJoCo simulation data. The joint position is obtained from the
    `qpos` array, which holds the generalized positions for all joints in the model.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose position is being retrieved.

    Returns
    ----------
    np.ndarray
        An array containing the position (or angle) of the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - The returned array includes the position values corresponding to the joint's degrees of freedom.
    - Ensure that the joint_name is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )
    q_indxs = get_joint_qpos_indxs(data, model, joint_name)
    return data.qpos[q_indxs]


def get_joint_dq(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[int, str]
) -> np.ndarray:
    """
    Retrieves the velocity of a joint in the MuJoCo simulation.

    This function extracts the current velocity of a specified joint from the
    MuJoCo simulation data. The joint's velocity is obtained from the `qvel` array,
    which holds the generalized velocities for all joints in the model.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose velocity is being retrieved.

    Returns
    ----------
    np.ndarray
        An array containing the velocity values of the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - The returned array includes the velocity values corresponding to the joint's degrees of freedom.
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )
    dq_indxs = get_joint_dof_indxs(model, joint_name)
    return data.qvel[dq_indxs]


def get_joint_ddq(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[int, str]
) -> np.ndarray:
    """
    Retrieves the acceleration of a joint in the MuJoCo simulation.

    This function extracts the current acceleration of a specified joint from the
    MuJoCo simulation data. The joint's acceleration is obtained from the `qacc` array,
    which holds the generalized accelerations for all joints in the model.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose acceleration is being retrieved.

    Returns
    ----------
    np.ndarray
        An array containing the acceleration values of the specified joint.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - The returned array includes the acceleration values corresponding to the joint's degrees of freedom.
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )
    ddq_indxs = get_joint_dof_indxs(model, joint_name)
    return data.qacc[ddq_indxs]


def get_joint_qpos_indxs(
    data: mj.MjModel, model: mj.MjModel, joint_name: Union[int, str]
) -> np.ndarray:
    """
    Retrieves the indices in the `qpos` array corresponding to the specified joint in the MuJoCo model.

    This function determines the indices in the `qpos` array where the position(s) of the specified joint are stored.
    The indices are computed based on the joint's position address and its dimension.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose `qpos` indices are to be retrieved.

    Returns
    ----------
    np.ndarray
        An array of indices in the `qpos` array that correspond to the specified joint.

    Notes
    -----
    - The `joint_name` can be provided as either a string (name) or an integer (ID).
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )
    addr = get_joint_qpos_addr(model, joint_name)
    joint_dim = get_joint_dim(data, model, joint_name)
    return list(range(addr, addr + joint_dim))


def get_joint_dof_indxs(model: mj.MjModel, joint_name: Union[int, str]) -> np.ndarray:
    """
    Retrieves the degrees of freedom (DOF) indices for a specified joint in the MuJoCo model.

    This function obtains the indices of the degrees of freedom for a given joint, which represent
    the joint's DOF in the MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[int, str]
        The name or ID of the joint whose DOF indices are to be retrieved.

    Returns
    ----------
    np.ndarray
        An array of indices corresponding to the degrees of freedom of the specified joint.

    Notes
    -----
    - The `joint_name` can be provided as either a string (name) or an integer (ID).
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )
    jdof = model.jnt_dofadr[joint_name]
    if not isinstance(jdof, np.ndarray):
        jdof = np.array([jdof])
    return jdof


def get_joint_dim(
    data: mj.MjData, model: mj.MjModel, joint_name: Union[str, int]
) -> int:
    """
    Retrieves the dimensionality (number of `qpos` elements) of the specified joint in the MuJoCo simulation.

    This function determines the number of position elements (`qpos`) associated with a given joint,
    which reflects the joint's degrees of freedom or configuration.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : Union[str, int]
        The name or ID of the joint whose dimensionality is to be retrieved.

    Returns
    ----------
    int
        The number of `qpos` elements for the specified joint, representing its dimensionality.

    Notes
    -----
    - The `joint_name` can be specified as either a string (name) or an integer (ID).
    - Ensure that the `joint_name` is valid and corresponds to an existing joint in the model.
    """
    joint_name = (
        joint_name
        if isinstance(joint_name, int)
        else name2id(model, joint_name, ObjType.JOINT)
    )
    return len(data.joint(joint_name).qpos)


def load_keyframe(
    model: mj.MjModel,
    data: mj.MjData,
    keyframe_name: str,
    file_path: str,
    return_xml: bool = False,
    log: bool = True,
    step: bool = True,
) -> Optional[str]:
    """
    Load a MuJoCo simulation state from a specified keyframe in an XML file or string.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object which defines the simulation.
    data : mj.MjData
        The MuJoCo data object to load the state into.
    keyframe_name : str
        The name of the keyframe state to load.
    return_xml : bool, optional
        If True, return the XML content as a string instead of loading it into the model. Defaults to False.
    file_path : str, optional
        Path to the XML file to load from. If None, defaults to "keyframes/<file_name>.xml".
    log : bool, optional
        Whether to log the loading action. Defaults to True.
    step : bool, optional
        Whether to step the simulation after loading the state. Defaults to True.

    Returns
    -------
    Optional[str]
        If return_xml is True, returns the XML content as a string. Otherwise, returns None.

    Raises
    ------
    ValueError
        If the specified state_name does not correspond to a valid keyframe.
    FileNotFoundError
        If the specified file_path does not exist.
    """

    if return_xml:
        # Load XML content from file and return as string
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                xml_content = file.read()
            return xml_content
        else:
            print(f"File {file_path} not found.")
            return None

    # Proceed with loading state from file
    if os.path.exists(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        keyframe_element = root.find(".//keyframe")
        if keyframe_element is not None:
            try:
                state_id = model.keyframe(keyframe_name).id
            except Exception as e:
                raise ValueError(
                    f'"{keyframe_name}" does not seem to be available as a state, have you remembered to include the keyframes in your scene file? ERROR({e})'
                )

            mj.mj_resetDataKeyframe(model, data, state_id)
            if log:
                print(f'Loaded : "{keyframe_name}" keyframe')
    else:
        print(f"File {file_path} not found.")
    if step:
        mj.mj_step(model, data)


def save_keyframe(
    model: mj.MjModel,
    data: mj.MjData,
    keyframe_name: str,
    save_path: str,
) -> None:
    """
    Save the current state of a MuJoCo simulation to an XML file as a keyframe.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model object.
    data : mj.MjData
        The MuJoCo data object containing the current simulation state.
    keyframe_name : str
        The name to associate with the saved state.
    save_path : str
        The file path where the keyframes XML will be saved.

    Notes
    -----
    - If the file doesn't exist, it is created with a root `<mujoco>` tag.
    - If a keyframe with the same name already exists, it will be overwritten.
    """

    def format_array(array) -> str:
        """Format a NumPy array into a space-separated string."""
        return " ".join(map(str, np.asarray(array).flatten()))

    # Check if the file exists and load or initialize the XML
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        tree = ET.parse(save_path)
        root = tree.getroot()
    else:
        root = ET.Element("mujoco")
        tree = ET.ElementTree(root)

    # Find or create the <keyframe> element
    keyframe_element = root.find("keyframe")
    if keyframe_element is None:
        keyframe_element = ET.SubElement(root, "keyframe")

    # Check for an existing keyframe with the same name
    existing_key = keyframe_element.find(f"./key[@name='{keyframe_name}']")
    if existing_key is not None:
        key_element = existing_key
        print(f"[save_keyframe]: overwriting {key_element}")
    else:
        key_element = ET.SubElement(keyframe_element, "key")
        key_element.set("name", keyframe_name)

    # Update the keyframe attributes
    key_element.set("time", f"{data.time:.6f}")
    key_element.set("qpos", format_array(data.qpos))
    key_element.set("qvel", format_array(data.qvel))
    key_element.set("ctrl", format_array(data.ctrl))
    key_element.set("mpos", format_array(data.mocap_pos))
    key_element.set("mquat", format_array(data.mocap_quat))

    # Save the updated XML to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        rough_string = ET.tostring(root, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        file.write(reparsed.toprettyxml(indent="    "))

    print(f"Saved keyframe '{keyframe_name}' to '{save_path}'.")


def apply_wrench(
    data: mj.MjData,
    model: mj.MjModel,
    body_name: str,
    wrench: Union[np.ndarray, List, Tuple],
) -> None:
    """
    Applies a wrench (force and torque) to a specific body in the MuJoCo simulation.

    The wrench is a 6-dimensional vector that includes both force (fx, fy, fz) and torque (tx, ty, tz).
    This function allows you to apply the wrench to a body in the simulation using its name.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object associated with the current simulation state.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    body_name : str
        The name of the body to which the wrench will be applied.
    wrench : Union[np.ndarray, List, Tuple]
        The wrench to be applied, provided as a 6-dimensional vector in any of the following formats:
        - numpy array
        - list
        - tuple

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the wrench is not a 6-dimensional vector or if the body name does not exist in the model.
    TypeError
        If the wrench is not a type that can be converted to a numpy array.
    """

    # Ensure the wrench is a numpy array
    if isinstance(wrench, (list, tuple)):
        try:
            wrench = np.array(wrench)
        except Exception as e:
            raise TypeError(f"Unable to convert wrench to numpy array: {e}")

    # Validate the wrench dimensions
    if wrench.shape != (6,):
        raise ValueError(
            "The wrench must be a 6-dimensional vector [fx, fy, fz, tx, ty, tz]."
        )

    # Get the body ID
    try:
        target_id = name2id(model, body_name, ObjType.BODY)
    except Exception:
        raise ValueError(f"The body name '{body_name}' does not exist in the model.")

    # Apply the wrench to the specified body
    data.xfrc_applied[target_id, :] = wrench


def get_geoms_in_contact(data: mj.MjData, model: mj.MjModel) -> List[Tuple[str, str]]:
    """
    Retrieve a list of geometry pairs currently in contact in the MuJoCo simulation.

    This function iterates through all contacts in the MuJoCo simulation data and
    collects the names of geometries that are in contact. It returns a list of tuples,
    where each tuple contains the names of two geometries that are in contact.

    Args
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.

    Returns
    ----------
    List[Tuple[str, str]]:
        A list of tuples, where each tuple contains the names of two geometries that are in contact.

    Notes
    ----------
    Ensure that the MuJoCo model and data are correctly initialized before calling this function.
    The function assumes that all contact pairs are relevant and does not filter them.
    """
    contact_states = []

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = id2name(model, contact.geom1, ObjType.GEOM)
        geom2_name = id2name(model, contact.geom2, ObjType.GEOM)
        contact_states.append((geom1_name, geom2_name))

    return contact_states


def get_bodies_in_contact(data: mj.MjData, model: mj.MjModel) -> List[Tuple[str, str]]:
    geom_ids_in_contact = []

    for i in range(data.ncon):
        contact = data.contact[i]
        geom_ids_in_contact.append((contact.geom1, contact.geom2))

    gids = [gid for gid in geom_ids_in_contact]
    for gid in gids:
        print(
            "\t",
            id2name(model, gid[0], ObjType.GEOM),
            id2name(model, gid[1], ObjType.GEOM),
        )
    bids = [model.geom_bodyid[gid[0]] for gid in gids]

    print(f"{gids=}")
    print(f"{bids=}")

    body_names = [
        (
            id2name(model, int(model.geom_bodyid[gid[0]]), ObjType.BODY),
            id2name(model, int(model.geom_bodyid[gid[1]]), ObjType.BODY),
        )
        for gid in geom_ids_in_contact
    ]

    return body_names


def get_joint_pos(data: mj.MjData, model: mj.MjModel, joint_name: str) -> np.ndarray:
    """
    Retrieve the position of a specified joint in the MuJoCo simulation.

    This function returns the anchor position of the specified joint.

    Args:
    ----------
    data : mj.MjData
        The MuJoCo data object containing simulation data.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    joint_name : str
        The name of the joint whose position is to be retrieved.

    Returns:
    ----------
    np.ndarray:
        The anchor position of the specified joint as a NumPy array.

    Notes:
    ----------
    Ensure that the specified joint name exists in the MuJoCo model.
    Raises a ValueError if the joint name is not found.
    """
    try:
        joint_id = name2id(model, joint_name, ObjType.JOINT)
        return data.joint(joint_id).xanchor
    except Exception as e:
        raise ValueError(f"Joint '{joint_name}' not found in the model.") from e


def get_geom_distance(
    data: mj.MjData,
    model: mj.MjModel,
    geom1: Union[int, str],
    geom2: Union[int, str],
    distmax: float = 10.0,
) -> Tuple[float, np.ndarray]:
    """
    Calculate the smallest signed distance between two geometries (geoms) and the segment from one geom to the other.

    This function computes the distance between two specified geoms and provides the segment vector between them.
    The distance is computed within a specified maximum distance, `distmax`. If the distance is greater than
    `distmax`, the function will return `distmax`.

    Args:
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation.
    model : mj.MjModel
        The MuJoCo model object containing the simulation model.
    geom1 : Union[int, str]
        The ID or name of the first geometry.
    geom2 : Union[int, str]
        The ID or name of the second geometry.
    distmax : float, optional
        The maximum distance for the distance calculation. Defaults to 10.0.

    Returns:
    ----------
    Tuple[float, np.ndarray]
        A tuple where the first element is the smallest signed distance between the two geoms, and the second
        element is the segment vector from `geom1` to `geom2`.

    Raises:
    ----------
    ValueError
        If `geom1` or `geom2` is not a valid geometry ID or name, or if `distmax` is not a positive number.

    Notes:
    ----------
    - Ensure that both `geom1` and `geom2` are valid geometry IDs or names within the MuJoCo model.
    - The distance calculation is sensitive to the model and simulation state. Ensure that the simulation is updated
      if necessary before calling this function.
    """

    if not isinstance(distmax, (int, float)) or distmax <= 0:
        raise ValueError(
            f"Invalid `distmax` value: {distmax}. It must be a positive number."
        )

    # Convert geom names to IDs if necessary
    geom1 = name2id(model, geom1, ObjType.GEOM) if isinstance(geom1, str) else geom1
    geom2 = name2id(model, geom2, ObjType.GEOM) if isinstance(geom2, str) else geom2

    # Initialize the segment vector
    from_to = np.zeros(6)

    # Calculate the distance and segment
    distance = mj.mj_geomDistance(model, data, geom1, geom2, distmax, from_to)

    _from = from_to[:3]
    _to = from_to[3:]

    return distance, _from, _to


@dataclass
class ContactState:
    """
    A class to represent the state of a contact in a MuJoCo simulation.

    Attributes
    ----------
    H : np.ndarray
        A 36-element cone Hessian, set by `mj_updateConstraint`.
    dim : int
        The contact space dimensionality: 1, 3, 4, or 6.
    dist : float
        The distance between the nearest points; negative values indicate penetration.
    efc_address : int
        The address in the constraint force Jacobian.
    elem : np.ndarray
        A 2-element array of integers representing element IDs; -1 for geom or flex vertex.
    exclude : int
        Exclusion flag for the contact: 0 (include), 1 (in gap), 2 (fused), 3 (no dofs).
    flex : np.ndarray
        A 2-element array of integers representing flex IDs; -1 for geom.
    frame : np.ndarray
        A 9-element array representing the contact frame. The normal is in [0-2] and points from geom[0] to geom[1].
    friction : np.ndarray
        A 5-element array representing friction parameters: tangent1, 2, spin, roll1, roll2.
    geom : np.ndarray
        A 2-element array of integers representing the IDs of the geometries in contact; -1 for flex.
    geom1 : int
        The first geometry index.
    geom2 : int
        The second geometry index.
    geom_names : np.ndarray
        A 2-element array of strings representing the names of the geometries in contact.
    geom1_name : str
        The name of the first geometry.
    geom2_name : str
        The name of the second geometry.
    includemargin : float
        The inclusion margin for the contact; includes if dist < includemargin = margin - gap.
    mu : float
        The coefficient of friction.
    pos : np.ndarray
        A 3-element array representing the position of the contact point, typically the midpoint between geometries.
    solimp : np.ndarray
        A 5-element array for solver impedance parameters.
    solref : np.ndarray
        A 2-element array for solver reference parameters for the normal direction.
    solreffriction : np.ndarray
        A 2-element array for solver reference parameters related to friction directions.
    vert : np.ndarray
        A 2-element array of integers representing vertex IDs; -1 for geom or flex element.
    index : int
        The index of the contact in the simulation.
    model : mj.MjModel
        The MuJoCo model object.
    data : mj.MjData
        The MuJoCo data object.
    c_wrench : np.ndarray
        A 6-element array representing the wrench (force and torque) at the contact in the contact frame.
    w_wrench : np.ndarray
        A 6-element array representing the wrench (force and torque) at the contact in the world frame.

    Methods
    -------
    _compute_wrench():
        Computes the wrench for the contact based on the model, data, and contact index.

    _transform_wrench_to_global(wrench: np.ndarray):
        Transforms the contact wrench from the contact frame to the world frame.

    from_mjcontact(cls, mjcontact, model, data, index=-1):
        Creates a ContactState instance from a given MjContact object.
    """

    data: mj.MjData = None
    model: mj.MjModel = None
    H: np.ndarray = field(default_factory=lambda: np.zeros(36))
    dim: int = 0
    dist: float = 0.0
    efc_address: int = 0
    elem: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    exclude: int = 0
    flex: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    frame: np.ndarray = field(default_factory=lambda: np.zeros(9))
    friction: np.ndarray = field(default_factory=lambda: np.zeros(5))
    geom: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    geom1: int = 0
    geom2: int = 0
    geom_names: np.ndarray = field(
        default_factory=lambda: np.array([-1, -1], dtype=int)
    )
    geom1_name: str = None
    geom2_name: str = None
    includemargin: float = 0.0
    mu: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    solimp: np.ndarray = field(default_factory=lambda: np.zeros(5))
    solref: np.ndarray = field(default_factory=lambda: np.zeros(2))
    solreffriction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vert: np.ndarray = field(default_factory=lambda: np.array([-1, -1], dtype=int))
    index: int = -1
    c_wrench: np.ndarray = field(init=False)
    w_wrench: np.ndarray = field(init=False)

    def __repr__(self):
        s = f"{type(self).__name__}(\n\t{self.pos=},\n\t{self.c_wrench=},\n\t{self.w_wrench=},\n\t{self.geom=},\n\t{self.geom_names=}\n)"
        return s

    def __post_init__(self):
        self.c_wrench = self._compute_wrench()
        self.w_wrench = self._transform_wrench_to_global(self.c_wrench)
        self.geom1_name = id2name(self.model, self.geom1, ObjType.GEOM)
        self.geom2_name = id2name(self.model, self.geom2, ObjType.GEOM)
        self.geom_names = np.array([self.geom1_name, self.geom2_name])

    def _compute_wrench(self):
        c_array = np.zeros(6, dtype=np.float64)
        if self.index >= 0 and self.model is not None and self.data is not None:
            mj.mj_contactForce(self.model, self.data, self.index, c_array)
        return c_array

    def _transform_wrench_to_global(self, wrench: np.ndarray):
        # Contact frame as 3x3 matrix
        contact_frame = self.frame.reshape(3, 3)
        w_wrench = np.ones_like(wrench)
        # Transform force (first 3 elements of wrench)
        w_wrench[:3] = contact_frame.T @ wrench[:3]
        w_wrench[3:] = contact_frame.T @ wrench[3:]
        return w_wrench

    @classmethod
    def from_mjcontact(
        cls, mjcontact, data: mj.MjData, model: mj.MjModel, index: int = -1
    ):
        return cls(
            H=mjcontact.H,
            dim=mjcontact.dim,
            dist=mjcontact.dist,
            efc_address=mjcontact.efc_address,
            elem=mjcontact.elem,
            exclude=mjcontact.exclude,
            flex=mjcontact.flex,
            frame=mjcontact.frame,
            friction=mjcontact.friction,
            geom=mjcontact.geom,
            geom1=mjcontact.geom1,
            geom2=mjcontact.geom2,
            includemargin=mjcontact.includemargin,
            mu=mjcontact.mu,
            pos=mjcontact.pos,
            solimp=mjcontact.solimp,
            solref=mjcontact.solref,
            solreffriction=mjcontact.solreffriction,
            vert=mjcontact.vert,
            index=index,
            data=data,
            model=model,
        )


class RobotInfo:
    def __init__(self, model: mj.MjModel, name: str):
        self.name = name
        self.m = model

        # Use list comprehensions to filter bodies and geometries by name
        self._body_ids = [
            model.body(i).id
            for i in range(model.nbody)
            if (self.name in model.body(i).name)
        ]
        self._body_names = [id2name(model, id, ObjType.BODY) for id in self._body_ids]

        self._geom_ids = [
            model.geom(i).id
            for i in range(model.ngeom)
            if (self.name in model.geom(i).name)
        ]
        self._geom_names = [id2name(model, id, ObjType.GEOM) for id in self._geom_ids]

        # Convert remaining function calls into list comprehensions
        self._joint_indxs = [
            model.joint(i).qposadr
            for i in range(model.njnt)
            if (self.name in model.joint(i).name)
        ]
        self._dof_indxs = [
            model.joint(i).dofadr
            for i in range(model.njnt)
            if (self.name in model.joint(i).name)
        ]
        self._joint_ids = [
            model.joint(i).id
            for i in range(model.njnt)
            if (self.name in model.joint(i).name)
        ]
        self._joint_names = [
            id2name(model, id, ObjType.JOINT) for id in self._joint_ids
        ]

        self._actuator_ids = [
            model.actuator(i).id
            for i in range(model.nu)
            if (self.name in model.actuator(i).name)
        ]
        self._actuator_names = [
            id2name(model, id, ObjType.ACTUATOR) for id in self._actuator_ids
        ]

        self._site_ids = [
            model.site(i).id
            for i in range(model.nsite)
            if (self.name in model.site(i).name)
        ]
        self._site_names = [id2name(model, id, ObjType.SITE) for id in self._site_ids]

        # Sensor information
        self._sensor_ids = [
            model.sensor(i).id
            for i in range(model.nsensor)
            if (self.name in model.sensor(i).name)
        ]
        self._sensor_names = [
            id2name(model, id, ObjType.SENSOR) for id in self._sensor_ids
        ]

        # Camera information
        self._camera_ids = [
            model.camera(i).id
            for i in range(model.ncam)
            if (self.name in model.camera(i).name)
        ]
        self._camera_names = [
            id2name(model, id, ObjType.CAMERA) for id in self._camera_ids
        ]

    def print(self) -> None:
        """
        Print the robot's information in a formatted and indented list.

        This method outputs the robot's detailed information, including base body name,
        number of bodies, actuators, joints, geometries, joint limits, and actuator limits.
        Each component's name and ID are listed with proper indentation to provide a clear,
        organized view of the robot's structure and components.
        """
        print(f"Robot Name: {self.name}")
        print(f"Number of Bodies: {len(self.body_names)}")
        print(f"Number of Actuators: {self.n_actuators}")
        print(f"Number of Joints: {self.n_joints}")
        print(f"Number of Geometries: {len(self.geom_names)}\n")

        print("Bodies:")
        for i, body_name in enumerate(self.body_names):
            print(f"  {i + 1}. {body_name} (ID: {self.body_ids[i]})")

        print("\nActuators:")
        for i, actuator_name in enumerate(self.actuator_names):
            print(f"  {i + 1}. {actuator_name} (ID: {self.actuator_ids[i]})")

        print("\nJoints:")
        for i, joint_name in enumerate(self.joint_names):
            print(f"  {i + 1}. {joint_name} (ID: {self.joint_ids[i]})")

        print("\nJoint Limits (min, max):")
        for i, (joint_name, limits) in enumerate(
            zip(self.joint_names, self.joint_limits.T)
        ):
            print(f"  {i + 1}. {joint_name}: {limits[0]:.2f}, {limits[1]:.2f}")

        print("\nActuator Limits (min, max):")
        for i, (actuator_name, limits) in enumerate(
            zip(self.actuator_names, self.actuator_limits.T)
        ):
            print(f"  {i + 1}. {actuator_name}: {limits[0]:.2f}, {limits[1]:.2f}")

        print("\nGeometries:")
        for i, geom_name in enumerate(self.geom_names):
            print(f"  {i + 1}. {geom_name} (ID: {self.geom_ids[i]})")

        print("\nSites:")
        for i, site_name in enumerate(self.site_names):
            print(f"  {i + 1}. {site_name} (ID: {self.site_ids[i]})")

        print("\nSensors:")
        for i, sensor_name in enumerate(self.sensor_names):
            print(f"  {i + 1}. {sensor_name} (ID: {self.sensor_ids[i]})")

        print("\nCameras:")
        for i, camera_name in enumerate(self.camera_names):
            print(f"  {i + 1}. {camera_name} (ID: {self.camera_ids[i]})")

    @property
    def body_ids(self) -> List[int]:
        """List of body IDs associated with the robot model."""
        return self._body_ids

    @body_ids.setter
    def body_ids(self, value: List[int]):
        self._body_ids = value

    @property
    def body_names(self) -> List[str]:
        """List of body names associated with the robot model."""
        return self._body_names

    @body_names.setter
    def body_names(self, value: List[str]):
        self._body_names = value

    @property
    def actuator_ids(self) -> List[int]:
        """List of actuator IDs associated with the robot model."""
        return self._actuator_ids

    @actuator_ids.setter
    def actuator_ids(self, value: List[int]):
        self._actuator_ids = value

    @property
    def actuator_names(self) -> List[str]:
        """List of actuator names associated with the robot model."""
        return self._actuator_names

    @actuator_names.setter
    def actuator_names(self, value: List[str]):
        self._actuator_names = value

    @property
    def geom_ids(self) -> List[int]:
        """List of geometry IDs associated with the robot model."""
        return self._geom_ids

    @geom_ids.setter
    def geom_ids(self, value: List[int]):
        self._geom_ids = value

    @property
    def geom_names(self) -> List[str]:
        """List of geometry names associated with the robot model."""
        return self._geom_names

    @geom_names.setter
    def geom_names(self, value: List[str]):
        self._geom_names = value

    @property
    def joint_indxs(self) -> List[int]:
        """List of indices for joint positions in the robot model."""
        return self._joint_indxs

    @joint_indxs.setter
    def joint_indxs(self, value: List[int]):
        self._joint_indxs = value

    @property
    def dof_indxs(self) -> List[int]:
        """List of indices for degrees of freedom in the robot model."""
        return self._dof_indxs

    @dof_indxs.setter
    def dof_indxs(self, value: List[int]):
        self._dof_indxs = value

    @property
    def joint_ids(self) -> List[int]:
        """List of joint IDs associated with the robot model."""
        return self._joint_ids

    @joint_ids.setter
    def joint_ids(self, value: List[int]):
        self._joint_ids = value

    @property
    def joint_names(self) -> List[str]:
        """List of joint names associated with the robot model."""
        return self._joint_names

    @joint_names.setter
    def joint_names(self, value: List[str]):
        self._joint_names = value

    @property
    def n_actuators(self) -> int:
        """Get the number of actuators."""
        return len(self.actuator_names)

    @property
    def n_joints(self) -> int:
        """Get the number of joints."""
        return len(self.joint_names)

    @property
    def joint_limits(self) -> np.ndarray:
        """Get the joint limits as a numpy array."""
        return np.array([self.m.joint(jn).range for jn in self.joint_names]).T

    @property
    def actuator_limits(self) -> np.ndarray:
        """Get the actuator limits as a numpy array."""
        return np.array([self.m.actuator(an).ctrlrange for an in self.actuator_names]).T

    @property
    def site_ids(self) -> List[int]:
        """List of site IDs associated with the robot model."""
        return self._site_ids

    @site_ids.setter
    def site_ids(self, value: List[int]):
        self._site_ids = value

    @property
    def site_names(self) -> List[str]:
        """List of site names associated with the robot model."""
        return self._site_names

    @site_names.setter
    def site_names(self, value: List[str]):
        self._site_names = value

    @property
    def sensor_ids(self) -> List[int]:
        """List of sensor IDs associated with the robot model."""
        return self._sensor_ids

    @sensor_ids.setter
    def sensor_ids(self, value: List[int]):
        self._sensor_ids = value

    @property
    def sensor_names(self) -> List[str]:
        """List of sensor names associated with the robot model."""
        return self._sensor_names

    @sensor_names.setter
    def sensor_names(self, value: List[str]):
        self._sensor_names = value

    @property
    def camera_ids(self) -> List[int]:
        """List of camera IDs associated with the robot model."""
        return self._camera_ids

    @camera_ids.setter
    def camera_ids(self, value: List[int]):
        self._camera_ids = value

    @property
    def camera_names(self) -> List[str]:
        """List of camera names associated with the robot model."""
        return self._camera_names

    @camera_names.setter
    def camera_names(self, value: List[str]):
        self._camera_names = value


def get_contact_states(
    data: mj.MjData,
    model: mj.MjModel,
    geom_names1: Union[int, str, List[Union[int, str]], np.ndarray] = None,
    geom_names2: Union[int, str, List[Union[int, str]], np.ndarray] = None,
) -> Tuple[bool, List[ContactState]]:
    """
    Retrieves the contact states for specified geometries in the MuJoCo simulation.

    Parameters
    ----------
    data : mj.MjData
        The MuJoCo data object containing the current state of the simulation, including contact information.
    model : mj.MjModel
        The MuJoCo model object, which defines the structure of the simulation including geometries.
    geom_names1 : Union[int, str, List[int], List[str], np.ndarray], optional
        The name or ID of the first geometry (or a list/array of them) to check for contact.
        If provided, the function filters the contact states to include only those involving this geometry.
        Default is None.
    geom_names2 : Union[int, str, List[int], List[str], np.ndarray], optional
        The name or ID of the second geometry (or a list/array of them) to check for contact.
        If provided, the function filters the contact states to include only those involving both specified geometries.
        Default is None.

    Returns
    -------
    Tuple[bool, List[ContactState]]
        - A boolean indicating whether any contact was found (`True` if there are contact states, `False` otherwise).
        - A list of ContactState objects representing the states of the contacts in the simulation.

    Notes
    -----
    - If both `geom_names1` and `geom_names2` are provided, the function returns only the contacts involving both geometries.
    - If only one geometry (either name or ID or list of them) is provided, the function returns all contacts involving that geometry.
    - If no geometry names or IDs are provided, the function returns all contact states in the simulation.

    Examples
    --------
    # Example 1: Get all contact states in the simulation
    in_contact, contact_states = get_contact_states(data, model)

    # Example 2: Get contact states involving a specific geometry by name
    in_contact, contact_states = get_contact_states(data, model, geom_names1='geom1')

    # Example 3: Get contact states involving two specific geometries by IDs
    in_contact, contact_states = get_contact_states(data, model, geom_names1=3, geom_names2=5)

    # Example 4: Get contact states involving any of multiple geometries
    in_contact, contact_states = get_contact_states(data, model, geom_names1=['geom1', 'geom2'])

    # Example 5: Get contact states involving pairs of geometries from two lists
    in_contact, contact_states = get_contact_states(data, model, geom_names1=[3, 5], geom_names2=['geom6', 'geom7'])
    """

    def get_geom_ids(geom_names):
        """Helper function to convert names/IDs to a list of geometry IDs."""
        if geom_names is None:
            return None
        if isinstance(geom_names, (int, str)):
            geom_names = [geom_names]  # Convert single value to list
        return [
            geom if isinstance(geom, int) else name2id(model, geom, ObjType.GEOM)
            for geom in geom_names
        ]

    geom_ids1 = get_geom_ids(geom_names1)
    geom_ids2 = get_geom_ids(geom_names2)

    # Get all contact states
    contact_states: List[ContactState] = []
    for i in range(data.ncon):
        contact = data.contact[i]
        cs = ContactState.from_mjcontact(contact, data, model, i)
        contact_states.append(cs)

    # Filter contacts based on the provided geometry names or IDs
    if geom_ids1 is not None:
        contact_states = [
            cs
            for cs in contact_states
            if cs.geom1 in geom_ids1 or cs.geom2 in geom_ids1
        ]

    if geom_ids2 is not None:
        contact_states = [
            cs
            for cs in contact_states
            if (cs.geom1 in geom_ids2 or cs.geom2 in geom_ids2)
            and (geom_ids1 is None or cs.geom1 in geom_ids1 or cs.geom2 in geom_ids1)
        ]

    in_contact = len(contact_states) != 0

    return in_contact, contact_states
