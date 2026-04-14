# Control Notes

## Current Control Split

- `config.ini` no longer carries continuum-chain Genesis `kp/kv` tuning.
- Commanded chain motion is applied kinematically via direct DOF position setting with optional rate limiting.
- Real Dynamixel PID gains are a separate hardware concern.

## Current Hardware Control

- Hardware command path is:
  - UI / commanded target
  - `bridge.py`
  - `engine/motor.py`
  - Dynamixel Goal Position
- Current hardware command path uses:
  - position control mode
  - profile velocity
  - profile acceleration
  - goal position
- Current code does not read or write Dynamixel Position P/I/D gains.

## Current Chain Modes

- `fixed`:
  - Genesis-fixed background / base part
- `commanded`:
  - theoretical chain motion driven by UI / IK targets
  - no model-state injection
- `simulated`:
  - chain pose is injected from model output
  - current placeholder model is direct hardware-state passthrough
  - future IMU / ArUco / camera fusion should replace that model

## About Dynamixel PID Values

- Dynamixel Wizard shows raw register values, not SI gains.
- Those values are model/control-table specific.
- If raw P/I/D values are brought later, they should be treated as hardware-only settings, not continuum sim settings.

## Future Work Reminder

When real motor PID values are available later:

1. Read actual Dynamixel control-table addresses for Position P/I/D.
2. Confirm motor model and register scale.
3. Treat those values as raw hardware gains.
4. Do not assume exact 1:1 conversion into any future simulation-side dynamics model.

## IK Direction Preference Note

- Current IK direction preference assumes world `+X`.
- Concretely, the solver prefers larger tip forward `x` component, i.e. maximizing `dot(tip_forward, [1, 0, 0])`.
- Later this should be generalized to an arbitrary preferred world-space direction vector:
  - `dir_score = dot(tip_forward, preferred_dir_world_unit)`
- That preferred direction may come from:
  - a fixed configured axis
  - a target-relative direction
  - a runtime-computed vector such as a wall normal / wall-facing vector
- When this is implemented, keep the lexicographic priority unchanged:
  - position feasibility first
  - then preferred direction alignment
  - then minimum path / posture cost from the real start pose

## Recent Hardware UI Behavior

- `use_hardware=true`:
  - sliders stay locked until `Torque On`
  - after `Disconnect Port` then `Apply Port`, startup pose sync is re-armed
- `use_hardware=false`:
  - hardware port / torque UI is hidden
