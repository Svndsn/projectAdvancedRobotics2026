# DMP Quick Guide

## What Is A DMP?

A Dynamic Movement Primitive (DMP) is a way to learn a motion from a recorded demonstration and replay a smooth version of that motion later.

In this project, the DMP is learned from the recorded TCP pose of the robot:
- position: `x, y, z`
- orientation: rotation vector

## Record A Demonstration

Run:

```bash
python -m sims.record_demo
```

In the viewer:
- double-click the orange sphere to select it
- drag it to move the robot
- press `SPACE` to start/stop recording
- press `ENTER` to save and exit

This saves a CSV in `data/demo/`.

## Replay The Demonstration With DMP

Run:

```bash
python -m sims.dmp_cartesian_ur5e --demo data/demo/your_file.csv --tau-scale 5
```

`--tau-scale 5` makes the learned motion slower. This gave the best result here.

When the replay finishes and the sim closes, it will automatically:
- save a replay log in `data/demo/`
- save plots in `data/plots/`

## What The Plots Mean

- `*_actual_path3d.png`: recorded TCP path vs DMP TCP path
- `*_actual_dmp_error.png`: error between recorded motion and DMP motion
- `*_position_compare.png`: DMP target vs executed replay
- `*_execution_error.png`: error between DMP target and executed replay
