# mj_sim

A repository to demonstrate, implement, test and learn control in MuJoCo.

## Table of Contents

- [mj\_sim](#mj_sim)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Getting Repository](#getting-repository)
    - [Setting up a Virtual Environment](#setting-up-a-virtual-environment)
    - [Activating a Virtual Environmentv](#activating-a-virtual-environmentv)
  - [Installing Dependencies](#installing-dependencies)
- [Real Time](#real-time)
- [Learning](#learning)
- [Docs](#docs)
- [VSCODE Compatibility](#vscode-compatibility)
- [License](#license)
- [Cite this work](#cite-this-work)

## Setup

### Getting Repository
To get the repository clone it into your directory of choice

```bash
git clone git@gitlab.sdu.dk:sdurobotics/teaching/mj_sim.git
```
then go into the repository by
```bash
cd mj_sim
```

### Setting up a Virtual Environment

To isolate the project's dependencies, it's recommended to use a virtual environment. If you haven't already **installed Python's venv module**, you can do so by running:

```bash
sudo apt install python3.12-venv   # For Linux (Debian/Ubuntu)
```


```bash
pip install virtualenv            # For Windows, assuming pip is installed (guide: https://phoenixnap.com/kb/install-pip-windows)
```

```bash
brew install virtualenv           # For MacOS
```

Once installed, you can **create a virtual environment** by executing the following commands:

```bash
python3.12 -m venv venv            # Linux (Ubuntu) and MacOS
```
```bash
virtualenv --python C:\Path\To\Python\python.exe venv            # Windows
```

### Activating a Virtual Environmentv
```bash
source venv/bin/activate          # Activate the virtual environment (Linux/Macos)
```
```bash
venv\Scripts\activate             # Activate the virtual environment (Windows)
```

## Installing Dependencies

After activating the virtual environment, you can install the project dependencies using pip. `mj_sim` uses [poetry](https://python-poetry.org/docs/) for managing dependencies and can thus be installed using
```bash
pip install poetry
```
Once poetry is installed in your virtual environment, install the project dependencies using 
```bash
poetry install
```
in the root of the project. Poetry will then install the project's base dependencies.

> [!WARNING]
> In case your poetry installation seems to run forever, kill the process and run the following 
> ```bash
> export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
> ```
> and try again.

Some modules requires more dependencies which can be installed in the following manner

 - **Learning** 
    ```bash
    poetry install --with learning
    ```
 - **Real**
    ```bash
    poetry install --with real
    ```
 - **All**
    ```bash
    poetry install --all-extras
    ```

# Real Time

Real time simulations can be found in [`sims/`](/sims/) and is run in the following manner as an example

```bash
python -m sims.opspace_ur5e  # Linux
mjpython -m sims.opspace_ur5e # MacOS
Python -m sims.opspace_ur5e   # Windows
```

A demo of this can be seen below

![here](/public/docs/real_time_sim_demo.gif)


# Learning

The learning uses JAX, Brax and MuJoCo for performing Deep Reinforcement Learning. See more [here](/learning/).

# Docs

A static website is generatd using [`pdoc`](https://pdoc.dev/) and can be found in `public/`. Access the documentation through 
```bash
<your-webbrowser-of-choise> public/index.html # e.g. firefox public/index.html
```
or online through the link found [here](https://sdurobotics.pages.sdu.dk/teaching/mj_sim/).

In case of generating pdoc documentation fails, remember to enable pdoc subprocess execution i.e.

```bash
export PDOC_ALLOW_EXEC=1
```

# VSCODE Compatibility

In case you are using [vscode](https://code.visualstudio.com/download) native type hinting is not a default for the MuJoCo or `ur_rtde` bindings (at the writing of this README.md). To add type hinting look here [here](https://github.com/google-deepmind/mujoco/issues/1292) (`pybind11-stubgen` is installed with the [`pyproject.toml`](pyproject.toml) when you install the dependencies through poetry) i.e. 
For MuJoCo:
```bash
pybind11-stubgen mujoco -o ./typings/
```
For `ur_thde`:
```bash
pybind11-stubgen rtde_control -o ./typings/
```
```bash
pybind11-stubgen rtde_receive -o ./typings/
```
```bash
pybind11-stubgen rtde_io -o ./typings/
```
For more documentation see [here](https://github.com/sizmailov/pybind11-stubgen).

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Cite this work
To cite this work using `bibtex` please use the following
```bibtex
@misc{staven2024mjsim,
  author       = {Staven, Victor M},
  title        = {mj\_sim},
  version      = {3.0.0},
  year         = {2024},
  howpublished = {\url{https://gitlab.sdu.dk/sdurobotics/teaching/mj_sim}},
  note         = {A repository to demonstrate, implement, test and learn control in MuJoCo.},
}
```
