# Robot Learning Lab Tasks

## Unitree G1 variants

Existing `Unitree-G1` task IDs continue to use the original 29DoF model. The
three-finger-hand variants are separate and contain `Unitree-G1-Dex3` in the
task ID:

- `RobotLab-Isaac-Velocity-Rough-Unitree-G1-Dex3-v0`
- `RobotLab-Isaac-Velocity-Flat-Unitree-G1-Dex3-v0`
- `RobotLab-Isaac-AMP-Flat-Unitree-G1-Dex3-v0`
- `RobotLab-MJLab-Velocity-Rough-Unitree-G1-Dex3`
- `RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3`
- `RobotLab-MJLab-AMP-Flat-Unitree-G1-Dex3`

The fixed 1 kg backpack copy has matching, separately registered task IDs:

- `RobotLab-Isaac-Velocity-Rough-Unitree-G1-Dex3-Backpack-v0`
- `RobotLab-Isaac-Velocity-Flat-Unitree-G1-Dex3-Backpack-v0`
- `RobotLab-Isaac-AMP-Flat-Unitree-G1-Dex3-Backpack-v0`
- `RobotLab-MJLab-Velocity-Rough-Unitree-G1-Dex3-Backpack`
- `RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3-Backpack`
- `RobotLab-MJLab-AMP-Flat-Unitree-G1-Dex3-Backpack`

Dex3 policies control all 43 joints. AMP reference observations stay on the
original 29 body joints because the current motion files contain no hand
trajectories.

Robot-learning environments shared across Isaac Lab and MJLab.

```text
robot_learning_lab_tasks/tasks/
  isaaclab/  # Direct and manager-based Isaac Lab environments
  mjlab/     # MJLab environments
```

Install only the backend you need:

```bash
uv pip install -e ".[isaaclab]"
uv pip install -e ".[mjlab]"
```

Importing `robot_learning_lab_tasks` does not import either simulator. Import a backend to register its environments:

```python
import robot_learning_lab_tasks.tasks.isaaclab
import robot_learning_lab_tasks.tasks.mjlab
```
