# From Onshape to a MuJoCo robot

You will design a small robot in Onshape (a box with two wheels), turn it into a MuJoCo model automatically,
and drive it in a simulation on the cluster. The cluster setup in `../README.md` must already work
(`python3 setup.py` all PASS).

## What is in this folder

| Path | What it is |
|---|---|
| `README.md` | This guide: design the robot in Onshape, export it, drive it. |
| `drive.py` | Drives the robot with a fixed plan (forward, spin, curve) and films it. |
| `.env.example` | Template for your Onshape keys. Copy it to `.env` and fill it in (`.env` stays out of git). |
| `onshape/` | The robot: `config.json` (which Onshape document), and the exported `robot.xml`, `scene.xml`, `assets/`. A hand-written sample robot is there until you export your own. |
| `sample/` | A copy of that sample robot, to fall back on. |
| `mjlab_task/` | Teach the robot to drive straight with reinforcement learning (own README). |
| `docs/` | How the Drive-Straight training works, in plain words. |

## First, a word about file formats

MuJoCo reads **MJCF**, its own XML format. **URDF** is a different XML format that comes from ROS (the robot
operating system). MuJoCo can also read URDF, but MJCF is the native one and what we use here.
The tool we use, [onshape-to-robot](https://onshape-to-robot.readthedocs.io/), can write either, and
we ask it for MJCF (`"output_format": "mujoco"` in `onshape/config.json`). It writes:

| File | What it is |
|---|---|
| `robot.xml` | the robot: bodies, joints (the wheel axles), motors, and the shapes |
| `scene.xml` | `robot.xml` plus a floor and a light (this is the file we load) |
| `assets/` | the 3D shapes (STL files) of your parts |

## Try the sample first (5 minutes)

`sample/` is a hand-written robot of the same kind, so you can see the whole pipeline work before touching Onshape.
In a laptop terminal in the `mujoco-hpc` folder:

```
python3 runIt.py --gpu --with 09_onshape_robot/sample 09_onshape_robot/drive.py sample/scene.xml
```

A video opens: the robot sits, drives straight, spins on the spot, then curves. `--with` copies the `sample`
folder to the cluster next to the script, and `sample/scene.xml` tells the script which model to load.

## Part 1: design the robot in Onshape

You need a free Onshape account (onshape.com; students can get the free Education plan). Create a new document.

**Part Studio** (draw the parts; units in millimetres):

1. **Chassis**: a box 200 mm (long, x) x 120 mm (wide, y) x 50 mm (tall, z). Rename the part `chassis`.
2. **Wheel**: a cylinder, 80 mm diameter, 20 mm thick, with its axis along the width (y) direction.
   Rename it `wheel`. You will use it twice. Use a **light material** (right-click the part > Material, for example ABS
   plastic): the exporter uses the mass Onshape computes, and a heavy steel robot needs much stronger motors.
3. **Skid**: a small sphere, 30 mm diameter. Rename it `skid`. A box on only two wheels tips over; the skid is the third
   point that holds it up and slides along the floor.

**Assembly** (put them together). Create an assembly named **`robot`**:

1. Insert `chassis` first and make sure it is **fixed** (right-click > Fix). It is the "trunk" the other parts hang from.
2. Insert `skid` and join it to the front underside of the chassis with a **Fastened mate** (so it never moves).
   Its lowest point must be level with the bottom of the wheels. With the wheel placement in step 3 the wheels
   reach 50 mm below the chassis centre, so the skid (15 mm radius) has its centre 35 mm below the chassis centre,
   which is 10 mm below the chassis underside.
3. Insert `wheel` twice. Put the wheels on the two sides of the chassis, with the axle line toward the **rear**
   (about 40 mm behind the chassis centre, 10 mm below the chassis centre), so the weight sits between the wheels and the skid.
4. Join each wheel to the chassis with a **Revolute mate** (the wheel turns about its axle). Select the wheel's
   mate connector first, then the chassis' point on the side where the axle goes. The mate's Z arrow is the
   axle: it must point along the wheel's axis.
5. **Name the two revolute mates `dof_left_wheel` and `dof_right_wheel`** (double-click the mate in the feature tree).
   The exporter turns a mate whose name starts with `dof_` into a joint, drops the `dof_`, and because the
   name contains `wheel` makes it a continuous (can spin forever) joint. So the joints are called `left_wheel` and
   `right_wheel`. If the wheel spins the wrong way when you drive it forwards, add `_inv` to the mate name
   (`dof_left_wheel_inv`) to flip it.

The exporter's own design notes cover more cases: <https://onshape-to-robot.readthedocs.io/en/latest/design.html>

## Part 2: export it

Do this once on your laptop.

1. Install the tool: `python3 -m pip install onshape-to-robot`
2. Make API keys. In Onshape: your account > Developer > API keys > Create new API key (tick read access). You get an
   **access key** and a **secret key**. **The secret is shown once, and it is a password: do not share it or paste it into
   a document, chat or public repository.**
3. Put the keys in a file named `.env` in this folder (`09_onshape_robot/`, the folder you run the tool from, not
   inside `onshape/`). Make it by copying the template, then open it and replace the two `PASTE_...` values with your
   access key and secret key (leave `ONSHAPE_API` as it is unless your school has its own Onshape address, such as
   `https://yourschool.onshape.com`):
   ```
   cp .env.example .env
   ```
   ```
   ONSHAPE_API=https://cad.onshape.com
   ONSHAPE_ACCESS_KEY=your-access-key
   ONSHAPE_SECRET_KEY=your-secret-key
   ```
   The file starts with a `.`, so Finder hides it; in VSCode it shows in the file list. A `.gitignore` here keeps `.env`
   out of git, but never share it anyway.
   (If the tool says it cannot find the keys, run these three as `export NAME=value` in your terminal instead.)
4. Open `onshape/config.json` and paste the browser address of your Onshape document (the tab with your assembly) in place of
   the placeholder. Leave `assembly_name` as `robot` if that is what you called the assembly.
5. Activate the Python virtual environment you installed `onshape-to-robot` into (skip this if you installed it
   system-wide), then from the `09_onshape_robot` folder run:
   ```
   source ../.venv/bin/activate
   onshape-to-robot onshape
   ```
   (On Windows the activate line is `..\.venv\Scripts\activate`.) Your prompt shows `(.venv)` when it is active; if you get
   `command not found: onshape-to-robot`, the environment is not active.
   `onshape/` now has `robot.xml`, `scene.xml` and `assets/` (replacing the sample `robot.xml` and `scene.xml` that
   were there). You will see `Unknown joint type: continuous`: that is harmless, the wheels are written as ordinary
   hinge joints with no angle limits. A warning `part wheel has no dynamics` means the wheel has no material in
   Onshape (right-click the part > Material); it still works, but give it a material for a realistic mass.
6. Open `onshape/robot.xml` in VSCode. Look for your two joints and their motors (`<actuator>`), for example:
   ```
   <joint name="left_wheel" .../>
   <position name="left_wheel" joint="left_wheel" .../>
   ```
   The names here are what `drive.py` needs.

## Part 3: drive it

1. `drive.py` loads `onshape/scene.xml` by default. If your motors are not called `left_wheel` and `right_wheel`
   (look in `robot.xml`), change the constants at the top of `drive.py`:
   ```python
   LEFT, RIGHT = "left_wheel", "right_wheel"    # the actuator names you found in robot.xml
   ```
2. From the `mujoco-hpc` folder:
   ```
   python3 runIt.py --gpu --with 09_onshape_robot/onshape 09_onshape_robot/drive.py
   ```
   The video (`drive.mp4`) opens when the job finishes. The script prints the joint and motor names it found, and
   where the robot is after each phase of the plan, in the job output.

Change `PLAN` in `drive.py` to make it do something else: each line is *(until this second, left speed, right speed)*
in radians per second. The wheel radius is 0.04 m, so 8 rad/s is about 0.3 m/s.

## If something goes wrong

| What you see | What to do |
|---|---|
| `onshape-to-robot` does not understand `config.json` | Versions differ. Check yours with `python3 -m pip show onshape-to-robot` and compare with the docs for that version: <https://onshape-to-robot.readthedocs.io/>. Older versions use `"document_id"` instead of a URL. |
| It says it cannot find the assembly | `assembly_name` must match the assembly's tab name in Onshape exactly. |
| Authentication error | Check the three values in `.env`; keys may need to be created with read access. |
| `ERROR: No Onshape API access key are set` | The tool cannot see `.env`. It must be in the folder you run `onshape-to-robot` from (`09_onshape_robot/`), not in `onshape/`. |
| `Invalid URL` | `config.json` still has the placeholder, or you did not save it. |
| `KeyError: Invalid name ... Valid names: [...]` | `LEFT` / `RIGHT` in `drive.py` do not match your motors. The error lists the right names. |
| Robot is pinned in place, or only the wheels turn | The exporter may have fixed the trunk to the world. `drive.py` frees it automatically; if it still will not move, check that the chassis is the fixed part in the assembly. |
| Wheels barely move | The robot is too heavy for the motors: choose a lighter material in Onshape (and export again), or raise the motor strength (`kp`, `forcerange`) in `robot.xml`. |
| It drives backwards, or the two wheels fight | A wheel's mate points the wrong way: add `_inv` to that mate's name and export again. |
| Front lifts off the floor when it starts moving, or it flips and tumbles | The motors are too strong for the robot: the wheel torque tips it over backwards. Lower `forcerange` (or `kp`) on the motors in `robot.xml`, or move the axle further back / the weight further forward. |
| Robot falls over | The skid is missing or in the wrong place. The weight (chassis centre) must sit between the axle and the skid. |

## What was tested

The whole path (Onshape design, export with onshape-to-robot 1.8.3, `drive.py` on the cluster) has been run end to end
on a box with two wheels and a skid. Send the newest file in `../logs/` if `runIt.py` fails.
