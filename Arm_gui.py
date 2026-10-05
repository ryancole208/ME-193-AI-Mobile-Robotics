# arm_gui.py -- runs on the PC (regular Python), e.g. from VS Code
# pip install matplotlib numpy
#
# Talks to the ESP32 over Wi-Fi (UDP), so Thonny can stay connected over USB.
# 1) In Thonny, run arm_esp32.py (it prints the IP it is listening on).
# 2) Join the ESP32's Wi-Fi network ("ArmDemo") from this computer.
# 3) Run this script.
#
# Coordinates: base at (0, 0), +y is straight up, +x is to the right (mm).
# Joint angles: 0 = pointing straight up, positive = leaning toward +x (clockwise).
# Joint 2 is measured relative to link 1.
#
# Controls (click the plot window first):
#   Left-click   set TARGET point (preview of both IK solutions + straight-line path)
#   Right-click  set the arm's CURRENT point (arm moves there via IK)
#   g            execute the planned trajectory on the real arm
#   e            toggle which IK solution is used (elbow left / elbow right)
#   1            reset arm to the straight-up start pose
#   x            stop / cancel motion
#   c            calibrate: press once, jog each joint with its encoder until it is
#                vertical, press again to save that as the 0 degree point
#
# Set ESP_IP = None to run in simulation mode (no hardware needed).

import math
import socket
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

# ---------------- Config ----------------
ESP_IP = "192.168.4.1"   # ESP32 in "ap" mode is always 192.168.4.1; in "sta" mode use the IP
                         # printed in Thonny. None = simulation mode.
UDP_PORT = 4210
L1 = 112.0           # mm: shoulder axis -> elbow axis  (130 - 18; adjust if different)
L2 = 112.0           # mm: elbow axis -> tip            (130 - 18; adjust if different)
# Used only in simulation mode. With a real arm, the limits are read from the ESP32
# (so the outline always matches what the firmware will actually allow).
SIM_LIMITS = ((-45.0, 45.0), (-135.0, 135.0))
# The real arm's command angles are mirrored about x = 0 relative to this GUI
# (a +theta command leans the arm toward -x). True = negate joint commands and limits
# going to/from the ESP32 so the target, path and arm all line up. Set False if the
# arm starts going to the mirror point again (e.g. after a firmware change).
MIRROR_X = True
TRAJ_STEPS = 60
TICK_MS = 40

R = L1 + L2 + 30     # plot half-width (mm)
CELL = 3.0           # workspace raster cell size (mm)
N = int(2 * R / CELL) + 1


# ---------------- Wi-Fi link (or simulation) ----------------
class Link:
    def __init__(self, ip):
        self.sock = None
        self.angles = [0.0, 0.0]
        self.limits = SIM_LIMITS
        self.calibrating = False
        self.last_rx = 0.0
        self.last_hello = 0.0
        if ip:
            try:
                self.addr = (ip, UDP_PORT)
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.setblocking(False)
                self._hello()
                print("Sending to", self.addr)
            except OSError as e:
                self.sock = None
                print("Socket error (%s) -> simulation mode" % e)
        else:
            print("Simulation mode")

    def _raw(self, data):
        try:
            self.sock.sendto(data, self.addr)
        except OSError:
            pass

    def _hello(self):
        self.last_hello = time.time()
        self._raw(b"H")

    def send(self, line):
        if self.sock:
            p = line.split(",")
            if MIRROR_X and p[0] == "S":
                line = "S,%.1f,%.1f" % (-float(p[1]), -float(p[2]))
            self._raw(line.encode())
        else:
            p = line.split(",")
            if p[0] == "S":
                self.angles = [float(p[1]), float(p[2])]
            elif p[0] == "R":
                self.angles = [0.0, 0.0]
            elif p[0] == "C":
                self.calibrating = not self.calibrating

    def poll(self):
        if self.sock:
            if time.time() - self.last_hello > 1.0:
                self._hello()                       # heartbeat keeps the link registered
            while True:
                try:
                    data, _ = self.sock.recvfrom(256)
                except OSError:
                    break
                p = data.decode(errors="ignore").strip().split(",")
                try:
                    if len(p) == 3 and p[0] == "A":
                        self.angles = [float(p[1]), float(p[2])]
                        if MIRROR_X:                # undo the mirror applied in send()
                            self.angles = [-self.angles[0], -self.angles[1]]
                        self.last_rx = time.time()
                    elif len(p) == 2 and p[0] == "M":
                        self.calibrating = (p[1].strip() == "1")
                        self.last_rx = time.time()
                    elif len(p) == 5 and p[0] == "L":
                        v = [float(s) for s in p[1:]]
                        if MIRROR_X:                # mirrored range: [lo, hi] -> [-hi, -lo]
                            v = [-v[1], -v[0], -v[3], -v[2]]
                        self.limits = ((v[0], v[1]), (v[2], v[3]))
                        self.last_rx = time.time()
                except ValueError:
                    pass
        return self.angles

    def status(self):
        if not self.sock:
            return "SIM"
        return "WIFI OK" if time.time() - self.last_rx < 1.5 else "WIFI: no signal"

    def close(self):
        if self.sock:
            self.sock.close()


# ---------------- Kinematics ----------------
def fk(t1, t2):
    """Forward kinematics. Angles in degrees from vertical (+ = toward +x); t2 is relative to link 1."""
    a = math.radians(t1)
    b = math.radians(t1 + t2)
    p1 = (L1 * math.sin(a), L1 * math.cos(a))
    p2 = (p1[0] + L2 * math.sin(b), p1[1] + L2 * math.cos(b))
    return (0.0, 0.0), p1, p2


def ik(x, y, elbow):
    """Inverse kinematics. elbow=+1 -> theta2>0 (elbow left), -1 -> theta2<0 (elbow right)."""
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1 + 1e-9:
        return None
    t2 = elbow * math.acos(max(-1.0, min(1.0, c2)))
    t1 = math.atan2(x, y) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
    t1 = (t1 + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(t1), math.degrees(t2)


def in_limits(s, lim):
    return (s is not None and lim[0][0] <= s[0] <= lim[0][1]
            and lim[1][0] <= s[1] <= lim[1][1])


def plan(start_xy, target_xy, elbow, lim):
    """Straight-line (Cartesian) path, solved with IK at every step."""
    pts = np.linspace(start_xy, target_xy, TRAJ_STEPS)
    sols = []
    for x, y in pts:
        s = ik(x, y, elbow)
        if not in_limits(s, lim):
            return pts, None, "path leaves the workspace near (%.0f, %.0f)" % (x, y)
        sols.append(s)
    return pts, sols, "ok"


def joint_move(a, b, n):
    return [tuple(v) for v in np.linspace(a, b, n)]


def workspace_mask(lim):
    """Raster of every tip position the arm can reach within its joint limits."""
    t1 = np.radians(np.arange(lim[0][0], lim[0][1] + 1e-9, 0.25))
    t2 = np.radians(np.arange(lim[1][0], lim[1][1] + 1e-9, 0.25))
    A, B = np.meshgrid(t1, t2)
    x = L1 * np.sin(A) + L2 * np.sin(A + B)
    y = L1 * np.cos(A) + L2 * np.cos(A + B)
    ix = np.floor((x + R) / CELL).astype(int)
    iy = np.floor((y + R) / CELL).astype(int)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
    m = np.zeros((N, N), dtype=bool)
    m[iy[ok], ix[ok]] = True
    return m


def edge_points(m):
    inner = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    edge = m[1:-1, 1:-1] & ~inner
    iy, ix = np.nonzero(edge)
    return -R + (ix + 1.5) * CELL, -R + (iy + 1.5) * CELL


# ---------------- App ----------------
class App:
    def __init__(self):
        self.link = Link(ESP_IP)
        self.limits = self.link.limits
        self.elbow = 1
        self.target = None
        self.queue = []
        self.hist = []
        self.msg = "Ready"

        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []         # free up keys like g / e / 1

        # Map on the left, status + math panel on the right (so nothing covers the map)
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.05,
                                   left=0.05, right=0.99, top=0.97, bottom=0.07)
        self.ax = self.fig.add_subplot(gs[0])
        self.info_ax = self.fig.add_subplot(gs[1])
        self.info_ax.axis("off")
        ax = self.ax
        cmap = ListedColormap([(0, 0, 0, 0), (0.2, 0.65, 0.3, 0.18)])
        self.ws_img = ax.imshow(np.zeros((N, N)), extent=[-R, -R + N * CELL, -R, -R + N * CELL],
                                origin="lower", cmap=cmap, vmin=0, vmax=1,
                                interpolation="nearest", zorder=0)
        self.ws_edge, = ax.plot([], [], ".", color="tab:green", ms=2.5, zorder=1,
                                label="feasible workspace")
        ax.set_xlim(-R, R)
        ax.set_ylim(-0.45 * R, R)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        self.trace, = ax.plot([], [], "-", color="tab:blue", alpha=0.35)
        self.path, = ax.plot([], [], "--", color="gray")
        self.ghost = {
            1: ax.plot([], [], "o-", color="tab:orange", lw=3, alpha=0.3)[0],
            -1: ax.plot([], [], "o-", color="tab:purple", lw=3, alpha=0.3)[0],
        }
        self.arm, = ax.plot([], [], "o-", color="tab:blue", lw=6, ms=10)
        self.tgt, = ax.plot([], [], "r*", ms=16)
        self.text = self.info_ax.text(0.02, 0.99, "", transform=self.info_ax.transAxes, va="top",
                                      family="monospace", fontsize=10,
                                      bbox=dict(facecolor="#f4f4f4", edgecolor="#cccccc"))
        self.math = self.info_ax.text(0.02, 0.78, "", transform=self.info_ax.transAxes, va="top",
                                      family="monospace", fontsize=10)
        self.banner = ax.text(0.5, 0.02, "", transform=ax.transAxes, ha="center", va="bottom",
                              color="red", fontsize=11, weight="bold",
                              bbox=dict(facecolor="white", alpha=0.85, edgecolor="red"))
        self.build_workspace()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.timer = self.fig.canvas.new_timer(interval=TICK_MS)
        self.timer.add_callback(self.tick)
        self.timer.start()

    def build_workspace(self):
        m = workspace_mask(self.limits)
        self.ws_img.set_data(m.astype(float))
        ex, ey = edge_points(m)
        self.ws_edge.set_data(ex, ey)

    # ----- actions -----
    def reset(self):
        if self.link.calibrating:
            self.msg = "Finish calibration first (press c)"
            return
        self.queue = []
        self.hist = []
        self.link.angles = [0.0, 0.0]
        self.link.send("R")
        self.msg = "Reset to straight-up start pose"

    def move_to(self, x, y):
        if self.link.calibrating:
            self.msg = "Finish calibration first (press c)"
            return
        s = ik(x, y, self.elbow)
        if not in_limits(s, self.limits):
            self.msg = "Can't reach (%.0f, %.0f) with this elbow setting" % (x, y)
            return
        self.hist = []
        self.queue = joint_move(self.link.angles, s, 40)
        self.msg = "Moving arm to (%.0f, %.0f)" % (x, y)

    def go(self):
        if self.link.calibrating:
            self.msg = "Finish calibration first (press c)"
            return
        if self.target is None:
            self.msg = "Set a target first (left-click)"
            return
        cur = fk(*self.link.angles)[2]
        _, sols, status = plan(cur, self.target, self.elbow, self.limits)
        if sols is None:
            self.msg = "Can't execute: " + status
            return
        self.hist = []
        self.queue = joint_move(self.link.angles, sols[0], 20) + [tuple(s) for s in sols]
        self.msg = "Executing trajectory..."

    def math_text(self, t1, t2, p0):
        """Step-by-step math for moving the tip from its current point p0 to the target."""
        if self.target is None:
            return "MOVE MATH\n\nLeft-click a target point to see the math\nfor moving the arm there."
        x0, y0 = p0
        x, y = self.target
        dx, dy = x - x0, y - y0
        d = math.hypot(dx, dy)
        side = "left, θ2>0" if self.elbow > 0 else "right, θ2<0"
        out = [
            "MOVE MATH   current → target   (elbow %s)" % side,
            "",
            "P0 = (%7.1f, %7.1f) mm   θ = (%6.1f°, %6.1f°)" % (x0, y0, t1, t2),
            "P1 = (%7.1f, %7.1f) mm" % (x, y),
            "",
            "1) Straight-line path",
            "   ΔP = P1 - P0 = (%.1f, %.1f)" % (dx, dy),
            "   |ΔP| = √(Δx² + Δy²) = %.1f mm" % d,
            "   P(s) = P0 + s·ΔP,  s = 0…1 in %d steps (%.1f mm/step)"
            % (TRAJ_STEPS, d / (TRAJ_STEPS - 1)),
            "",
            "2) Inverse kinematics at P1",
            "   r² = x² + y² = (%.1f)² + (%.1f)² = %.0f" % (x, y, x * x + y * y),
            "   cos θ2 = (r² - L1² - L2²) / (2·L1·L2)",
        ]
        c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        out.append("          = (%.0f - %.0f - %.0f) / %.0f = %.4f"
                   % (x * x + y * y, L1 * L1, L2 * L2, 2 * L1 * L2, c2))
        if abs(c2) > 1 + 1e-9:
            out += ["", "   |cos θ2| > 1  →  target is out of reach",
                    "   (r = %.1f mm, max reach L1+L2 = %.0f mm)" % (math.hypot(x, y), L1 + L2)]
            return "\n".join(out)

        n1, n2 = ik(x, y, self.elbow)
        b = math.radians(n2)
        k1 = L1 + L2 * math.cos(b)
        k2 = L2 * math.sin(b)
        a_deg = math.degrees(math.atan2(x, y))
        b_deg = math.degrees(math.atan2(k2, k1))
        out += [
            "   θ2 = %sacos(%.4f) = %.1f°" % ("+" if self.elbow > 0 else "-", c2, n2),
            "   k1 = L1 + L2·cos θ2 = %.1f" % k1,
            "   k2 = L2·sin θ2      = %.1f" % k2,
            "   θ1 = atan2(x, y) - atan2(k2, k1)",
            "      = %.1f° - (%.1f°) = %.1f°" % (a_deg, b_deg, n1),
            "",
            "3) Joint change",
            "   Δθ1 = %.1f° - (%.1f°) = %+.1f°" % (n1, t1, n1 - t1),
            "   Δθ2 = %.1f° - (%.1f°) = %+.1f°" % (n2, t2, n2 - t2),
            "",
            "4) Forward-kinematics check",
        ]
        _, q1, q2 = fk(n1, n2)
        out += [
            "   x = L1·sin θ1 + L2·sin(θ1+θ2) = %.1f + %.1f = %.1f"
            % (q1[0], q2[0] - q1[0], q2[0]),
            "   y = L1·cos θ1 + L2·cos(θ1+θ2) = %.1f + %.1f = %.1f"
            % (q1[1], q2[1] - q1[1], q2[1]),
            "",
        ]
        lim = self.limits
        if not in_limits((n1, n2), lim):
            out.append("   ✗ joint limits exceeded: θ1∈[%+.0f,%+.0f], θ2∈[%+.0f,%+.0f]"
                       % (lim[0][0], lim[0][1], lim[1][0], lim[1][1]))
        else:
            out.append("   ✓ within joint limits")
        return "\n".join(out)

    # ----- events -----
    def on_key(self, ev):
        if ev.key == "1":
            self.reset()
        elif ev.key == "e":
            self.elbow = -self.elbow
        elif ev.key == "g":
            self.go()
        elif ev.key == "x":
            self.queue = []
            self.msg = "Stopped"
        elif ev.key == "c":
            self.queue = []
            self.link.send("C")
            self.msg = "Toggling calibration..."

    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and tb.mode:
            return                      # pan/zoom tool active
        if ev.button == 1:
            self.target = (ev.xdata, ev.ydata)
            self.msg = "Target set - press g to move"
        elif ev.button == 3:
            self.move_to(ev.xdata, ev.ydata)

    # ----- main loop -----
    def tick(self):
        if self.link.calibrating:
            self.queue = []
        if self.queue:
            t1, t2 = self.queue.pop(0)
            self.link.send("S,%.1f,%.1f" % (t1, t2))
            if not self.queue:
                self.msg = "Done"
        t1, t2 = self.link.poll()

        if self.link.limits != self.limits:         # limits reported by the arm
            self.limits = self.link.limits
            self.build_workspace()

        _, p1, p2 = fk(t1, t2)
        self.arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
        self.hist.append(p2)
        self.hist = self.hist[-400:]
        self.trace.set_data([h[0] for h in self.hist], [h[1] for h in self.hist])

        if self.target is not None:
            tx, ty = self.target
            self.tgt.set_data([tx], [ty])
            feasible = 0
            for e, line in self.ghost.items():
                s = ik(tx, ty, e)
                if not in_limits(s, self.limits):
                    line.set_data([], [])
                    continue
                feasible += 1
                _, g1, g2 = fk(*s)
                line.set_data([0, g1[0], g2[0]], [0, g1[1], g2[1]])
                chosen = (e == self.elbow)
                line.set_alpha(0.9 if chosen else 0.25)
                line.set_linewidth(5 if chosen else 2)
            if not self.queue:
                pts, sols, status = plan(p2, self.target, self.elbow, self.limits)
                self.path.set_data(pts[:, 0], pts[:, 1])
                self.path.set_color("gray" if sols else "red")
                if feasible == 0:
                    self.msg = "Target is outside the feasible workspace"
                elif sols is None:
                    self.msg = "Preview: " + status + " (try 'e')"

        lim = self.limits
        self.banner.set_text(
            "CALIBRATING: turn each joint's encoder until the real arm is vertical,\n"
            "then press c to save it as the 0 degree point"
            if self.link.calibrating else "")
        elbow_name = "left (theta2>0)" if self.elbow > 0 else "right (theta2<0)"
        self.text.set_text(
            "link: %s   elbow: %s\n"
            "theta1=%6.1f  theta2=%6.1f deg\n"
            "limits: th1[%+.0f,%+.0f]  th2[%+.0f,%+.0f]\n"
            "tip=(%6.1f, %6.1f) mm\n"
            "%s\n\n"
            "L-click target | R-click move arm | g go\n"
            "e elbow | 1 reset | x stop | c calibrate"
            % (self.link.status(), elbow_name, t1, t2,
               lim[0][0], lim[0][1], lim[1][0], lim[1][1], p2[0], p2[1], self.msg)
        )
        self.math.set_text(self.math_text(t1, t2, p2))
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    app = App()
    plt.show()
    app.link.close()