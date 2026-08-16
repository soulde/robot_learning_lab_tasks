# Robot Learning Lab Tasks

Robot-learning environments shared across Isaac Lab and MJLab.

```text
robot_learning_lab_tasks/tasks/
  isaaclab/  # Direct and manager-based Isaac Lab environments
  mjlab/     # MJLab environments
```

Install only the backend you need:

```bash
pip install -e ".[isaaclab]"
pip install -e ".[mjlab]"
```

Importing `robot_learning_lab_tasks` does not import either simulator. Import a backend to register its environments:

```python
import robot_learning_lab_tasks.tasks.isaaclab
import robot_learning_lab_tasks.tasks.mjlab
```
