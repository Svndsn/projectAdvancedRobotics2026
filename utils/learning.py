from pathlib import Path
from typing import Callable, Dict, List, Tuple

import jax
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from brax import envs
from brax.io import model
from brax.training.acme.running_statistics import RunningStatisticsState
from jax import numpy as jp

from utils.args import Args


def create_data_directory(environment_name: str, session_name: str) -> Path:
    """
    Create a directory structure for storing data related to a specific environment and session.

    Args:
        environment_name (str): The name of the environment.
        session_name (str): The name of the session.

    Returns:
        Path: The path to the created session directory.
    """
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "learning" / "data"
    session_dir = data_dir / environment_name / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = session_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created directories for session: {session_dir}")
    return session_dir


def get_dimensions_from_params(
    params: Tuple[RunningStatisticsState, Dict],
) -> Tuple[int, int]:
    """
    Extract observation and action dimensions from model parameters.

    Args:
        params (Tuple[RunningStatisticsState, dict]):
            A tuple containing the running statistics state and a dictionary of model parameters.

    Returns:
        Tuple[int, int]: A tuple where the first element is the observation dimension (o_dim)
                         and the second element is the action dimension (a_dim).
    """
    parameter_dict = params[1]["params"]
    hidden_keys: List[str] = list(parameter_dict.keys())
    hidden_first, hidden_last = hidden_keys[0], hidden_keys[-1]

    # Observation dimension
    o_dim = np.array(parameter_dict[hidden_first]["kernel"]).shape[0]

    # Action dimension (last layer output divided by 2)
    a_dim = np.array(parameter_dict[hidden_last]["kernel"]).shape[1] // 2

    return o_dim, a_dim


def generate_demo_video(
    args: Args,
    make_inference_fn: Callable,
    params: tuple,
    session_path: Path,
    n_steps: int = 500,
    render_every: int = 2,
    camera_name: str = "side",
) -> list[np.ndarray]:
    """
    Generates a demonstration video of a reinforcement learning agent's performance.

    This function creates a video by simulating an environment and rendering frames
    at specified intervals. The agent's actions are determined using a provided
    inference function and trained model parameters.

    Args:
        args (Args): Command-line arguments or configuration for the environment.
        make_inference_fn (Callable): A function that generates an inference function
                                       from model parameters.
        params (tuple): Placeholder for model parameters (unused; parameters are loaded
                        from the session path).
        session_path (Path): Path to the directory containing the trained model parameters.
        n_steps (int): Number of simulation steps to run. Defaults to 500.
        render_every (int): Frequency of rendering frames (every `render_every` steps).
                            Defaults to 2.

    Returns:
        list[np.ndarray]: A list of rendered video frames, where each frame is a NumPy
                          array representing the image in RGB format.
    """
    params = model.load_params(session_path / "model")
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    eval_env = envs.get_environment(args.env_name)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(0)
    mj_model = eval_env.sys.mj_model
    mj_data = mj.MjData(mj_model)

    renderer = mj.Renderer(mj_model)
    ctrl = jp.zeros(mj_model.nu)

    frames = []
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)

        obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
        ctrl, _ = jit_inference_fn(obs, act_rng)

        mj_data.ctrl = ctrl
        # for _ in range(eval_env._n_frames):

        mj.mj_step(mj_model, mj_data)  # Physics step using MuJoCo mj_step.

        if i % render_every == 0:
            renderer.update_scene(mj_data, camera=camera_name)
            frames.append(renderer.render())

    frames = np.transpose(np.array(frames), axes=(0, 3, 1, 2))
    return frames
