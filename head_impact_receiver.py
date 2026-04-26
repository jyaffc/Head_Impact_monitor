# =============================================================================
# head_impact_receiver.py
# Python BLE Receiver - Head Impact Monitor
#
# AXIS CALIBRATION WIZARD:
#   Instead of guessing a hardcoded axis remap, the script learns which chip
#   axis corresponds to roll/pitch/yaw by watching the sensor respond to three
#   known physical movements. Click "Calibrate Axes" in the window and follow
#   the four-step prompts:
#       Step 0: Hold still    -> captures baseline quaternion
#       Step 1: Pitch up      -> identifies the pitch axis and sign
#       Step 2: Yaw left      -> identifies the yaw axis and sign
#       Step 3: Roll left     -> identifies the roll axis and sign
#   After capture the mapping is applied automatically for the rest of the session.
#
# REQUIRED LIBRARIES:
#   pip install bleak numpy matplotlib

import asyncio
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D subplot projection
import csv
import datetime
import threading
import time
from bleak import BleakClient, BleakScanner

# =============================================================================
# CONFIGURATION - Must match HeadImpactMonitor.ino exactly
# =============================================================================

BLE_DEVICE_NAME  = "HeadImpactMonitor"
SERVICE_UUID     = "12345678-1234-1234-1234-123456789abc"
CHAR_UUID_IMU    = "abcd1234-ab12-ab12-ab12-abcdef123456"  # NOTIFY: 44-byte IMU stream
CHAR_UUID_PLAYER = "abcd1234-ab12-ab12-ab12-abcdef123459"  # READ:   player ID string
CHAR_UUID_CMD    = "abcd1234-ab12-ab12-ab12-abcdef12345a"  # WRITE:  command to ESP32

# '<' = little-endian, '10f' = ten 32-bit floats, 'I' = one uint32
PACKET_FORMAT = "<10fI"
PACKET_SIZE   = struct.calcsize(PACKET_FORMAT)  # 44 bytes

# Impact detection threshold in m/s^2
# BNO055 removes gravity so this is purely dynamic linear acceleration.
# 14.7 m/s^2 = 1.5g
IMPACT_THRESHOLD_MS2 = 14.7

# =============================================================================
# HEAD MODEL DIMENSIONS
# X = left-right, Y = front-back, Z = up-down
# =============================================================================

HEAD_WIDTH_CM  = 15.0
HEAD_DEPTH_CM  = 19.0
HEAD_HEIGHT_CM = 23.0

HEAD_RADII = np.array([
    HEAD_WIDTH_CM  / 2.0,
    HEAD_DEPTH_CM  / 2.0,
    HEAD_HEIGHT_CM / 2.0
])

# =============================================================================
# SHARED STATE - written by BLE thread, read by render thread
# All access must go through data_lock
# =============================================================================

latest_quaternion   = np.array([1.0, 0.0, 0.0, 0.0])  # Identity = no rotation
latest_linear_accel = np.array([0.0, 0.0, 0.0])
latest_timestamp_ms = 0
impact_history      = []
data_lock           = threading.Lock()
running             = True
ble_connected       = False

# Player identification
player_name     = "Player1"
esp32_player_id = "Unknown"

# Software orientation zero reference
# q_corrected = conjugate(ref_quaternion) * q_raw
ref_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

# Sensor placement spherical coordinates
sensor_az     = 0.0   # Azimuth: 0=front, +pi/2=right, -pi/2=left
sensor_el     = 0.0   # Elevation: 0=ear-level, pi/2=top
sensor_placed = False
placement_mode = False

# =============================================================================
# PENDING BLE COMMAND
#
# FIX: Previously, on_recal_esp32 spawned a second BleakClient connection to
# send the RECAL write. The ESP32 BLE stack supports only one connection at a
# time, so that second connection would silently drop the streaming connection
# from ble_receiver, causing permanent loss of IMU data with no recovery.
#
# The fix: commands are now queued here under data_lock. The existing
# ble_receiver connection checks this flag each loop tick and sends the write
# through the already-open connection -- no second client, no dropped stream.
# =============================================================================

pending_ble_command = None  # Set by main thread, consumed by ble_receiver loop

# =============================================================================
# AXIS CALIBRATION STATE
#
# The calibration wizard learns the chip-to-model axis mapping empirically.
# For each of three movements (pitch up, yaw left, roll left), it computes the
# delta quaternion from the baseline and finds which of [qx, qy, qz] changed
# most. That component index is the chip axis for that motion. The sign of the
# change determines whether the mapping needs a negation.
#
# After calibration, remap_quaternion() uses axis_map (a list of 3 tuples):
#   axis_map[0] = (chip_index, sign) for model X (roll)
#   axis_map[1] = (chip_index, sign) for model Y (pitch)
#   axis_map[2] = (chip_index, sign) for model Z (yaw)
# =============================================================================

# Calibration steps:
#   0 = idle (not started)
#   1 = waiting to capture baseline (hold still)
#   2 = waiting to capture pitch up
#   3 = waiting to capture yaw left
#   4 = waiting to capture roll left
#   5 = complete
CAL_STEP_LABELS = [
    "",
    "STEP 1 of 4: Hold the sensor STILL, then click Capture",
    "STEP 2 of 4: Pitch head UP ~30 degrees, then click Capture",
    "STEP 3 of 4: Yaw head LEFT ~30 degrees, then click Capture",
    "STEP 4 of 4: Roll head LEFT ~30 degrees, then click Capture",
    "Calibration complete!"
]

cal_step        = 0                    # Current calibration step (0=not started)
cal_baseline_q  = None                 # Quaternion captured at Step 1 (hold still)
cal_captures    = [None, None, None]   # Quaternions captured at Steps 2/3/4
axis_map        = None                 # Learned mapping: list of (chip_idx, sign) x3
axis_calibrated = False                # True once calibration is complete


# =============================================================================
# QUATERNION MATH
# =============================================================================

def quaternion_to_rotation_matrix(q):
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.
    R satisfies: v_rotated = R @ v_original

    Parameters:
        q (np.ndarray): [w, x, y, z]
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    w, x, y, z = q
    xx=x*x; yy=y*y; zz=z*z
    xy=x*y; xz=x*z; yz=y*z
    wx=w*x; wy=w*y; wz=w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)]
    ])


def quaternion_multiply(q1, q2):
    """
    Hamilton product q1 * q2.
    Represents: apply q2 first, then q1. Not commutative.

    Parameters:
        q1, q2 (np.ndarray): [w, x, y, z]
    Returns:
        np.ndarray: product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_conjugate(q):
    """
    Conjugate (inverse) of a unit quaternion.
    conjugate([w, x, y, z]) = [w, -x, -y, -z]

    Parameters:
        q (np.ndarray): [w, x, y, z]
    Returns:
        np.ndarray: conjugate quaternion
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def apply_reference_quaternion(q_raw, q_ref):
    """
    Express q_raw relative to q_ref:
        q_corrected = conjugate(q_ref) * q_raw

    When q_ref equals q_raw at the moment of zeroing, q_corrected becomes
    identity (no rotation), making the current physical pose the neutral pose.

    Parameters:
        q_raw (np.ndarray): current IMU quaternion [w, x, y, z]
        q_ref (np.ndarray): reference quaternion [w, x, y, z]
    Returns:
        np.ndarray: corrected quaternion [w, x, y, z]
    """
    return quaternion_multiply(quaternion_conjugate(q_ref), q_raw)


def remap_quaternion(q_raw):
    """
    Reorder and sign-flip the chip-frame quaternion vector components to match
    the head model coordinate frame (X=roll, Y=pitch, Z=yaw).

    If axis calibration has been completed, uses the empirically learned mapping
    stored in axis_map. Each entry is (chip_index, sign) where chip_index is
    0/1/2 corresponding to qx/qy/qz in the raw quaternion vector part.

    If calibration has not been run yet, returns the raw quaternion unchanged
    so the model at least displays something until the user calibrates.

    Parameters:
        q_raw (np.ndarray): raw quaternion from BNO055 [w, x, y, z]
    Returns:
        np.ndarray: remapped quaternion [w, x, y, z]
    """
    if not axis_calibrated or axis_map is None:
        return q_raw  # No calibration yet - pass through raw

    # q_raw[1:] = [qx, qy, qz] are the vector components
    # axis_map[i] = (chip_index, sign) for model axis i (0=roll, 1=pitch, 2=yaw)
    vec = q_raw[1:]  # chip vector part: indices 0=qx, 1=qy, 2=qz

    remapped_x = axis_map[0][1] * vec[axis_map[0][0]]  # model roll
    remapped_y = axis_map[1][1] * vec[axis_map[1][0]]  # model pitch
    remapped_z = axis_map[2][1] * vec[axis_map[2][0]]  # model yaw

    return np.array([q_raw[0], remapped_x, remapped_y, remapped_z])


def compute_axis_map(baseline_q, pitch_q, yaw_q, roll_q):
    """
    Determine the chip-to-model axis mapping from four calibration quaternions.

    For each physical gesture (pitch, yaw, roll), compute the delta quaternion
    relative to the baseline. For small rotations, the vector part of the delta
    quaternion is approximately proportional to the axis of rotation. The
    component with the largest absolute value is the chip axis that corresponds
    to that physical motion. Its sign determines whether a negation is needed.

    delta_q = conjugate(baseline_q) * gesture_q
    dominant_index = argmax(|delta_q[1:3]|)
    sign = +1 if delta_q[dominant_index+1] > 0 else -1

    The physical gestures define the model axes as:
        pitch up  -> model Y axis (front-back tilt)
        yaw left  -> model Z axis (vertical rotation)
        roll left -> model X axis (side tilt)

    Parameters:
        baseline_q (np.ndarray): quaternion while holding still [w,x,y,z]
        pitch_q    (np.ndarray): quaternion after pitching up   [w,x,y,z]
        yaw_q      (np.ndarray): quaternion after yawing left   [w,x,y,z]
        roll_q     (np.ndarray): quaternion after rolling left  [w,x,y,z]
    Returns:
        list: [(chip_idx, sign), ...] for model axes [roll, pitch, yaw]
              chip_idx is 0/1/2 corresponding to qx/qy/qz
    """
    results = {}

    # gesture_name -> (gesture_q, model_axis_name)
    gestures = [
        ("pitch", pitch_q, "pitch"),
        ("yaw",   yaw_q,   "yaw"),
        ("roll",  roll_q,  "roll"),
    ]

    for name, g_q, model_axis in gestures:
        # Compute delta quaternion: how much did we rotate from baseline?
        delta_q = quaternion_multiply(quaternion_conjugate(baseline_q), g_q)

        # Vector part [qx, qy, qz] of delta_q indicates the rotation axis.
        # For a pure single-axis rotation, only one component will be large.
        vec = delta_q[1:]

        # Find which chip axis responded most strongly
        chip_idx = int(np.argmax(np.abs(vec)))

        # Sign: positive delta means the user's movement goes in the positive
        # direction for that chip axis. We want the model to also move positively
        # for that gesture (pitch up = +Y, yaw left = +Z, roll left = +X).
        # Since all three gestures are defined as positive model directions,
        # a positive delta means no negation needed; negative means negate.
        sign = 1 if vec[chip_idx] > 0 else -1

        results[model_axis] = (chip_idx, sign)
        print(f"[CAL] {name}: chip qxyz[{chip_idx}] sign={sign:+d}  "
              f"(delta vec = [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}])")

    # Return in model axis order: [roll (X), pitch (Y), yaw (Z)]
    return [results["roll"], results["pitch"], results["yaw"]]


# =============================================================================
# ELLIPSOID GEOMETRY
# =============================================================================

def build_ellipsoid(radii, resolution=24):
    """
    Generate ellipsoid surface vertex meshgrids via spherical parameterization.
        X = rx * sin(phi) * cos(theta)
        Y = ry * sin(phi) * sin(theta)
        Z = rz * cos(phi)
    phi in [0, pi], theta in [0, 2*pi]

    Parameters:
        radii      (np.ndarray): [rx, ry, rz] semi-axes in cm
        resolution (int): angular grid density
    Returns:
        tuple: (X, Y, Z) meshgrid arrays shape (resolution x resolution)
    """
    phi, theta = np.meshgrid(
        np.linspace(0, np.pi, resolution),
        np.linspace(0, 2*np.pi, resolution)
    )
    X = radii[0] * np.sin(phi) * np.cos(theta)
    Y = radii[1] * np.sin(phi) * np.sin(theta)
    Z = radii[2] * np.cos(phi)
    return X, Y, Z


def rotate_ellipsoid(X, Y, Z, R):
    """
    Apply 3x3 rotation matrix R to all ellipsoid vertices simultaneously.
    Flattens grids to (3, N^2), applies R, reshapes back.

    Parameters:
        X, Y, Z (np.ndarray): vertex grids from build_ellipsoid()
        R (np.ndarray): 3x3 rotation matrix
    Returns:
        tuple: (Xr, Yr, Zr) rotated grids in same shape
    """
    shape = X.shape
    verts   = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    verts_r = R @ verts
    return (verts_r[0].reshape(shape),
            verts_r[1].reshape(shape),
            verts_r[2].reshape(shape))


def sensor_position_3d(az, el, radii):
    """
    Convert spherical placement angles to a 3D point on the ellipsoid surface.
    +X=right, +Y=front, +Z=up. az=0 is front, az=+pi/2 is right.

    Parameters:
        az    (float): azimuth in radians
        el    (float): elevation from equatorial plane in radians
        radii (np.ndarray): [rx, ry, rz] semi-axes in cm
    Returns:
        np.ndarray: [x, y, z] in cm
    """
    x = radii[0] * np.cos(el) * np.sin(az)
    y = radii[1] * np.cos(el) * np.cos(az)
    z = radii[2] * np.sin(el)
    return np.array([x, y, z])


# =============================================================================
# IMPACT LOGGING HELPERS
# =============================================================================

def impact_direction_label(dx, dy, dz):
    """
    Convert an acceleration unit vector to a human-readable anatomical label
    matching the direction strings shown on the NeuroTrack dashboard
    (Frontal, Rear, Left lateral, Right lateral, Crown, Chin, Oblique).

    Coordinate system matches the head model (X=right, Y=front, Z=up).
    The unit vector points in the direction the impact force came from,
    so a frontal hit has a large positive Y component.

    Strategy: check the dominant axis first (whichever absolute component
    is largest). If it is clearly dominant (> 0.55) use the clean cardinal
    label. If the top two components are close together, use an oblique label.

    Parameters:
        dx, dy, dz (float): components of the acceleration unit vector
    Returns:
        str: human-readable impact direction label
    """
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    dominant = max(ax, ay, az)

    # Crown / Chin (vertical axis dominates)
    if az == dominant:
        if az > 0.55:
            return "Crown" if dz > 0 else "Chin"
        if ay >= ax:
            return "Frontal-Crown" if dy > 0 else "Rear-Crown"
        return "Lateral-Crown"

    # Frontal / Rear (front-back axis dominates)
    if ay == dominant:
        if ay > 0.55:
            return "Frontal" if dy > 0 else "Rear"
        if ax >= az:
            return "Frontal" if dy > 0 else "Rear"
        return "Frontal-Crown" if dy > 0 else "Rear-Crown"

    # Left / Right lateral (side axis dominates)
    if ax > 0.55:
        return "Right lateral" if dx > 0 else "Left lateral"

    # All three components roughly equal -> diffuse oblique
    return "Oblique"


def impact_confidence(magnitude_g):
    """
    Derive a confidence score (0-100) estimating how likely an event is a
    genuine head impact rather than sensor noise or an artifact.

    Higher magnitude events are more likely to be true impacts. The BNO055
    fusion output is reliable at high accelerations, and large amplitudes are
    statistically unlikely to come from walking noise or minor handling.

    Mapping (linear ramp, hard-capped at 95):
        1.5 g  (detection threshold)  ->  60%
        3.0 g                         ->  71%
        5.0 g                         ->  84%
        7.5 g                         ->  95%  (cap)

    Formula: confidence = clamp(60 + (magnitude_g - 1.5) * 10, 60, 95)

    Parameters:
        magnitude_g (float): net linear acceleration magnitude in g
    Returns:
        float: confidence percentage rounded to one decimal place
    """
    threshold_g = IMPACT_THRESHOLD_MS2 / 9.81   # 1.5 g at default threshold
    raw = 60.0 + (magnitude_g - threshold_g) * 10.0
    return round(min(95.0, max(60.0, raw)), 1)


# =============================================================================
# IMPACT LOGGING
# =============================================================================

def save_impact_log(p_name, esp_id, s_az, s_el):
    """
    Save impact events to a timestamped CSV compatible with the NeuroTrack
    coach dashboard upload parser (lib/session-upload-analysis.ts).

    Two files are written per session:
      impact_log_<player>_<ts>.csv   -- clean data table for the dashboard
      impact_log_<player>_<ts>.json  -- session metadata sidecar (debug info)

    Root cause of previous parser failures:
    PapaParse (the JavaScript CSV parser used by the dashboard) always treats
    ROW 1 as the column header row, regardless of content. The previous CSV
    had a metadata block (session_id, player_python, etc.) as the first rows,
    so the parser read "session_id" as a column name and never saw playerName,
    eventIntensity, or timestamp at all. Fix: CSV now starts with the column
    header on line 1 with no rows above it. All metadata moves to a JSON
    sidecar file with the same base name.

    CSV column names match the exact strings the NeuroTrack parser checks:
      playerName     -> athlete identity mapping
      timestamp      -> session ordering
      eventIntensity -> alert scoring in g
      load           -> cumulative exposure (= eventIntensity because the
                        BNO055 does not output a separate cumulative load value)

    Parameters:
        p_name (str): player name entered at Python startup prompt
        esp_id (str): player ID string read from ESP32 BLE characteristic
        s_az   (float): sensor azimuth placement angle in radians
        s_el   (float): sensor elevation placement angle in radians
    """
    if not impact_history:
        print("[LOG] No impacts recorded this session.")
        return

    import json

    ts_str    = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"impact_log_{p_name}_{ts_str}"
    csv_file  = f"{base_name}.csv"
    json_file = f"{base_name}.json"

    # ------------------------------------------------------------------
    # JSON sidecar - session metadata kept OUT of the CSV so that row 1
    # of the CSV can be the column header without anything above it.
    # ------------------------------------------------------------------
    metadata = {
        "session_id":    f"{p_name}_{ts_str}",
        "player_python": p_name,
        "player_esp32":  esp_id,
        "sensor_az_deg": round(np.degrees(s_az), 1),
        "sensor_el_deg": round(np.degrees(s_el), 1),
        "threshold_ms2": IMPACT_THRESHOLD_MS2,
        "threshold_g":   round(IMPACT_THRESHOLD_MS2 / 9.81, 3),
        "axis_map":      str(axis_map) if axis_map is not None else None,
        "total_events":  len(impact_history)
    }
    with open(json_file, mode="w") as jf:
        json.dump(metadata, jf, indent=2)
    print(f"[LOG] Metadata: {json_file}")

    # ------------------------------------------------------------------
    # CSV - row 1 is the column header, data rows follow immediately.
    # Nothing appears before row 1. PapaParse reads row 1 as the header;
    # every subsequent row is a data record. This is the fix.
    # ------------------------------------------------------------------
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)

        # Row 1: column headers - exact strings the NeuroTrack parser checks.
        # "direction" and "confidence" are derived columns added to match the
        # anatomical labels and confidence scores shown on the dashboard.
        writer.writerow([
            "playerName",      # athlete identity - exact string parser checks
            "playerId",        # ESP32 BLE characteristic ID (secondary field)
            "timestamp",       # ISO 8601 event time - exact string parser checks
            "timestamp_ms",    # ESP32 millis() since power-on (raw device time)
            "eventIntensity",  # net linear accel in g - exact string parser checks
            "load",            # cumulative exposure proxy = eventIntensity/event
            "direction",       # anatomical label (Frontal, Crown, Left lateral, etc.)
            "confidence",      # estimated impact confidence % (60-95, magnitude-derived)
            "magnitude_ms2",   # same value in m/s^2 (reference only)
            "direction_x",     # raw unit vector X for advanced analysis
            "direction_y",     # raw unit vector Y
            "direction_z"      # raw unit vector Z
        ])

        # Rows 2+: one row per detected impact event
        for ev in impact_history:
            intensity_g = ev["magnitude_g"]                        # float, used twice
            dx, dy, dz  = ev["direction"]                          # unit vector components
            direction   = impact_direction_label(dx, dy, dz)       # anatomical label
            confidence  = impact_confidence(intensity_g)           # % score

            writer.writerow([
                p_name,
                esp_id,
                ev["datetime"],             # goes into "timestamp" column
                ev["timestamp_ms"],
                f"{intensity_g:.3f}",       # eventIntensity in g
                f"{intensity_g:.3f}",       # load (same value - no separate load sensor)
                direction,                  # e.g. "Frontal", "Left lateral", "Crown"
                f"{confidence:.1f}",        # e.g. "73.5"
                f"{ev['magnitude_ms2']:.3f}",
                f"{dx:.4f}",
                f"{dy:.4f}",
                f"{dz:.4f}"
            ])

    print(f"[LOG] Saved:    {csv_file}  ({len(impact_history)} events)")


# =============================================================================
# BLE NOTIFICATION CALLBACK
# Called by bleak ~100 Hz in the BLE background thread.
# =============================================================================

def on_imu_notification(sender, raw_data):
    """
    Unpack 44-byte BLE IMU packet, apply axis remap, update shared state.

    Parameters:
        sender   : BLE characteristic handle (unused, required by bleak API)
        raw_data (bytearray): 44-byte packet from ESP32
    """
    global latest_quaternion, latest_linear_accel, latest_timestamp_ms
    global impact_history

    if len(raw_data) != PACKET_SIZE:
        print(f"[WARN] Bad packet: {len(raw_data)} bytes (expected {PACKET_SIZE})")
        return

    # Unpack little-endian: 10 x float32, 1 x uint32
    (qw, qx, qy, qz,
     linAccX, linAccY, linAccZ,
     gyroX, gyroY, gyroZ,
     timestamp_ms) = struct.unpack(PACKET_FORMAT, raw_data)

    # Apply empirical axis remap learned from calibration wizard.
    # If not yet calibrated, remap_quaternion returns raw values unchanged.
    q_remapped = remap_quaternion(np.array([qw, qx, qy, qz]))

    accel_vec = np.array([linAccX, linAccY, linAccZ])
    accel_mag = np.linalg.norm(accel_vec)

    # Impact detection on net dynamic acceleration magnitude
    if accel_mag > IMPACT_THRESHOLD_MS2:
        direction = accel_vec / accel_mag if accel_mag > 0 else accel_vec
        event = {
            "datetime":      datetime.datetime.now().isoformat(),
            "timestamp_ms":  timestamp_ms,
            "magnitude_ms2": accel_mag,
            "magnitude_g":   accel_mag / 9.81,
            "direction":     direction
        }
        with data_lock:
            impact_history.append(event)
        print(f"[IMPACT] {player_name} | {accel_mag:.2f} m/s^2 "
              f"({accel_mag/9.81:.2f}g) | "
              f"dir=[{direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}]")

    with data_lock:
        latest_quaternion   = q_remapped
        latest_linear_accel = accel_vec
        latest_timestamp_ms = timestamp_ms


# =============================================================================
# BLE RECEIVER (async, runs in background daemon thread)
#
# FIX: This function now contains an outer retry loop so that if the connection
# drops for any reason (disconnect, timeout, Windows WinRT BLE error) it waits
# briefly and re-scans automatically instead of dying permanently.
#
# The pending_ble_command flag is also checked here each tick so that RECAL
# (and any future commands) are sent through this already-open connection,
# eliminating the need to ever open a second BleakClient.
# =============================================================================

async def ble_receiver():
    """
    Scan for ESP32, connect, subscribe to IMU notifications, and auto-reconnect.

    Uses explicit client.connect() / client.disconnect() instead of async with
    BleakClient. The async with pattern triggers GATT discovery inside __aenter__
    and a client.disconnect() inside __aexit__. On Windows WinRT, this can cause
    an immediate apparent disconnect: __aenter__ completes, the body runs for a
    moment, then __aexit__ fires its cleanup disconnect before the loop has done
    anything useful. The explicit pattern separates connection, session work, and
    teardown into distinct phases so we know exactly which step failed.

    Key additions vs. the previous version:
    - disconnected_callback + asyncio.Event so hardware-initiated disconnects
      break out of the inner loop immediately (no waiting for next sleep tick).
    - 500ms pause between scan completion and connect: on Windows, the WinRT
      advertisement watcher needs time to fully release the radio before a GATT
      connection attempt. Without this the connect either fails or drops instantly.
    - timeout passed to client.connect() not BleakClient constructor (the
      constructor timeout parameter was removed in bleak 0.21+).
    - Separate try/except for connect, start_notify, and session loop so each
      failure prints a specific message instead of a generic "Connection lost."
    """
    global running, ble_connected, esp32_player_id, pending_ble_command

    while running:

        # ---- Scan phase ----------------------------------------------------
        print(f"[BLE] Scanning for '{BLE_DEVICE_NAME}'...")
        device = None
        try:
            device = await BleakScanner.find_device_by_name(
                BLE_DEVICE_NAME, timeout=15.0
            )
        except Exception as e:
            print(f"[BLE] Scan error: {e}")

        if device is None:
            print(f"[BLE] '{BLE_DEVICE_NAME}' not found. "
                  "Is the ESP32 powered on and advertising? Retrying in 3s...")
            await asyncio.sleep(3.0)
            continue

        print(f"[BLE] Found: {device.name} ({device.address})")

        # Windows WinRT fix: the BLE advertisement watcher spun up by
        # BleakScanner.find_device_by_name needs ~500ms to fully release the
        # radio before a GATT connection attempt. Without this pause the
        # connection either fails outright or drops within the first second.
        await asyncio.sleep(0.5)

        # ---- Connect phase -------------------------------------------------
        # The disconnect_event is set by on_hw_disconnect when the ESP32 drops
        # the connection from its side (power loss, out of range, RECAL reset,
        # etc). The inner loop checks this flag so it exits immediately instead
        # of waiting for the next asyncio.sleep(0.1) tick to notice the drop.
        disconnect_event = asyncio.Event()

        def on_hw_disconnect(client):
            # Called by bleak from the asyncio thread when the BLE link drops.
            # Setting an asyncio.Event here is safe because bleak fires this
            # callback from within the same event loop that runs ble_receiver.
            print("[BLE] Hardware disconnect callback fired.")
            disconnect_event.set()

        # Instantiate client with disconnect callback but do NOT connect yet.
        # timeout is passed to connect() below, not the constructor (bleak 0.21+).
        client = BleakClient(device, disconnected_callback=on_hw_disconnect)

        try:
            await client.connect(timeout=20.0)
        except Exception as e:
            print(f"[BLE] connect() failed: {e}. Retrying in 3s...")
            try:
                await client.disconnect()
            except Exception:
                pass
            await asyncio.sleep(3.0)
            continue

        print(f"[BLE] Connected. MTU: {client.mtu_size} bytes")

        # ---- Session phase -------------------------------------------------
        # All GATT operations and the streaming loop run here. Any exception
        # inside falls through to the finally block which always disconnects.
        session_started = False
        try:
            # One-time READ of player ID characteristic
            try:
                raw_id = await client.read_gatt_char(CHAR_UUID_PLAYER)
                esp32_player_id = raw_id.decode("utf-8").strip()
                print(f"[BLE] ESP32 Player ID: '{esp32_player_id}'")
            except Exception as e:
                print(f"[WARN] Could not read player characteristic: {e}")
                esp32_player_id = "Unknown"

            # Subscribe to IMU NOTIFY stream.
            # Separated into its own try so a failure here prints a specific
            # message rather than the generic session error below.
            try:
                await client.start_notify(CHAR_UUID_IMU, on_imu_notification)
            except Exception as e:
                print(f"[BLE] start_notify failed: {e}")
                print("  -> Check that BLE2902 descriptor is added in ESP32 sketch.")
                # Fall through to finally; no inner loop runs.
                raise  # Re-raise so the outer except catches it too

            ble_connected = True
            session_started = True
            print(f"[BLE] Streaming | threshold: {IMPACT_THRESHOLD_MS2} m/s^2 "
                  f"({IMPACT_THRESHOLD_MS2/9.81:.2f}g)\n")

            # ---- Inner keep-alive and command dispatch loop ----------------
            # Exits on: (a) running = False (window closed), or
            #           (b) disconnect_event set (ESP32 dropped the link).
            while running and not disconnect_event.is_set():
                # Check for a command queued by the main thread.
                # Read and clear under the lock so it only fires once.
                with data_lock:
                    cmd = pending_ble_command
                    pending_ble_command = None

                if cmd is not None:
                    try:
                        await client.write_gatt_char(
                            CHAR_UUID_CMD, cmd.encode("utf-8")
                        )
                        print(f"[CMD] Sent via existing connection: '{cmd}'")
                    except Exception as e:
                        print(f"[CMD] Write failed: {e}")

                await asyncio.sleep(0.1)

            # Unsubscribe cleanly only if WE exited the loop (not a hardware drop)
            if not disconnect_event.is_set():
                try:
                    await client.stop_notify(CHAR_UUID_IMU)
                except Exception:
                    pass  # Already disconnected is fine here

        except Exception as e:
            if session_started:
                # Exception thrown during the streaming loop itself
                print(f"[BLE] Session error: {e}")
            # If not session_started, the error was already printed above

        finally:
            # Always reset the connected flag and disconnect the client,
            # regardless of whether the session ran or failed during setup.
            ble_connected = False
            try:
                await client.disconnect()
            except Exception:
                pass  # Already disconnected is fine

        # ---- Decide retry delay based on how the session ended -------------
        if disconnect_event.is_set():
            print("[BLE] ESP32 dropped the connection. Reconnecting in 2s...")
        elif not session_started:
            print("[BLE] Session setup failed. Retrying in 3s...")
        elif running:
            print("[BLE] Session ended normally. Restarting in 2s...")

        if running:
            delay = 3.0 if not session_started else 2.0
            await asyncio.sleep(delay)

    print("[BLE] Receiver stopped.")


def ble_thread_func():
    """Entry point for background BLE daemon thread."""
    asyncio.run(ble_receiver())


def send_command_async(cmd_str):
    """
    Queue a BLE command to be sent on the next tick of the ble_receiver loop.

    FIX: Previously this spawned a new thread that opened a second BleakClient
    to send the command. The ESP32 BLE stack supports one connection at a time,
    so that second connection would drop the streaming connection from
    ble_receiver, causing permanent loss of IMU data.

    Now it just sets pending_ble_command under data_lock. The existing
    ble_receiver coroutine reads it and sends the write through its already-open
    connection, which keeps the IMU stream alive.

    Parameters:
        cmd_str (str): ASCII command string (e.g. "RECAL")
    """
    global pending_ble_command
    with data_lock:
        pending_ble_command = cmd_str
    print(f"[CMD] Command queued for existing connection: '{cmd_str}'")


# =============================================================================
# 2D PLACEMENT PANEL SETUP
# =============================================================================

def setup_placement_panels(ax_front, ax_top, radii):
    """
    Draw head silhouette ellipses in the Front View (XZ) and Top View (XY) panels.

    Parameters:
        ax_front : matplotlib Axes for front view
        ax_top   : matplotlib Axes for top view
        radii    (np.ndarray): [rx, ry, rz] semi-axes in cm
    """
    rx, ry, rz = radii

    for ax in (ax_front, ax_top):
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
        ax.set_aspect('equal')

    ax_front.set_title("Front View  (click to place sensor)",
                        color='#cccccc', fontsize=8, pad=4)
    ax_front.set_xlabel("X  left <- -> right  (cm)", color='#888888', fontsize=7)
    ax_front.set_ylabel("Z  down <- -> up  (cm)",    color='#888888', fontsize=7)
    ax_front.set_xlim(-rx*1.45, rx*1.45)
    ax_front.set_ylim(-rz*1.45, rz*1.45)
    ax_front.add_patch(mpatches.Ellipse(
        (0, 0), width=2*rx, height=2*rz,
        edgecolor='#8899bb', facecolor='#1a2035', linewidth=1.5, linestyle='--'
    ))
    ax_front.axhline(0, color='#333355', linewidth=0.7, linestyle=':')
    ax_front.axvline(0, color='#333355', linewidth=0.7, linestyle=':')
    ax_front.text( rx*1.3,  0,       'R',    color='#778899', fontsize=7, ha='center', va='center')
    ax_front.text(-rx*1.3,  0,       'L',    color='#778899', fontsize=7, ha='center', va='center')
    ax_front.text( 0,        rz*1.3, 'Top',  color='#778899', fontsize=7, ha='center', va='center')
    ax_front.text( 0,       -rz*1.3, 'Chin', color='#778899', fontsize=7, ha='center', va='center')

    ax_top.set_title("Top View  (updates azimuth only)",
                     color='#cccccc', fontsize=8, pad=4)
    ax_top.set_xlabel("X  left <- -> right  (cm)", color='#888888', fontsize=7)
    ax_top.set_ylabel("Y  back <- -> front  (cm)", color='#888888', fontsize=7)
    ax_top.set_xlim(-rx*1.45, rx*1.45)
    ax_top.set_ylim(-ry*1.45, ry*1.45)
    ax_top.add_patch(mpatches.Ellipse(
        (0, 0), width=2*rx, height=2*ry,
        edgecolor='#8899bb', facecolor='#1a2035', linewidth=1.5, linestyle='--'
    ))
    ax_top.axhline(0, color='#333355', linewidth=0.7, linestyle=':')
    ax_top.axvline(0, color='#333355', linewidth=0.7, linestyle=':')
    ax_top.text( rx*1.3,  0,      'R',     color='#778899', fontsize=7, ha='center', va='center')
    ax_top.text(-rx*1.3,  0,      'L',     color='#778899', fontsize=7, ha='center', va='center')
    ax_top.text( 0,  ry*1.3,     'Front',  color='#778899', fontsize=7, ha='center', va='center')
    ax_top.text( 0, -ry*1.3,     'Back',   color='#778899', fontsize=7, ha='center', va='center')


# =============================================================================
# MAIN RENDER LOOP (main thread only - matplotlib requirement)
# =============================================================================

def run_render_loop():
    """
    Build the matplotlib figure and run the animation render loop.

    Layout (GridSpec 2 rows x 2 cols):
        Col 0 (wide, spans both rows): 3D rotating head model
        Col 1 Row 0: Front View 2D placement panel
        Col 1 Row 1: Top View 2D placement panel

    Buttons (bottom of figure):
        Calibrate Axes   - opens the axis calibration wizard
        Capture          - advances the calibration wizard one step
        Zero Orientation - software re-zero of the reference pose
        Place Sensor     - activates click-to-place mode on 2D panels
        Recal ESP32      - sends RECAL command via existing BLE connection
    """
    global running, ref_quaternion
    global sensor_az, sensor_el, sensor_placed, placement_mode
    global cal_step, cal_baseline_q, cal_captures, axis_map, axis_calibrated

    # Build base (unrotated) ellipsoid vertex grids once
    X0, Y0, Z0 = build_ellipsoid(HEAD_RADII, resolution=24)

    # ---- Figure layout -----------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[3, 1.7], height_ratios=[1, 1],
        hspace=0.50, wspace=0.28,
        left=0.04, right=0.97, top=0.92, bottom=0.12
    )

    ax3d     = fig.add_subplot(gs[:, 0], projection='3d')
    ax_front = fig.add_subplot(gs[0, 1])
    ax_top   = fig.add_subplot(gs[1, 1])

    ax3d.set_facecolor('#1a1a2e')

    max_dim = float(max(HEAD_RADII)) * 1.4
    ax3d.set_xlim(-max_dim, max_dim)
    ax3d.set_ylim(-max_dim, max_dim)
    ax3d.set_zlim(-max_dim, max_dim)
    ax3d.set_xlabel('X  (roll)',  color='#cccccc', labelpad=8)
    ax3d.set_ylabel('Y  (pitch)', color='#cccccc', labelpad=8)
    ax3d.set_zlabel('Z  (yaw)',   color='#cccccc', labelpad=8)
    ax3d.tick_params(colors='#888888')
    ax3d.set_title(f'Head Impact Monitor  --  {player_name}',
                   color='white', fontsize=11, pad=14)

    # ---- Text overlays (figure coordinates, not axes) ----------------------
    status_text = fig.text(0.03, 0.97, 'Connecting to ESP32...',
                           color='#ffcc44', fontsize=9,
                           verticalalignment='top', fontfamily='monospace')
    impact_text = fig.text(0.03, 0.94, 'Impacts: 0  |  Player: loading...',
                           color='#ff9966', fontsize=9,
                           verticalalignment='top', fontfamily='monospace')

    # Calibration wizard instruction line - shown during calibration steps
    cal_text = fig.text(0.03, 0.91, '',
                        color='#ffff44', fontsize=9, fontweight='bold',
                        verticalalignment='top', fontfamily='monospace')

    # Placement mode hint line
    place_hint = fig.text(0.03, 0.88, '',
                          color='#aaffaa', fontsize=8,
                          verticalalignment='top', fontfamily='monospace')

    # World-space reference axes (fixed, do not rotate with head)
    al = max_dim * 0.9
    ax3d.plot([-al, al], [0, 0], [0, 0], color='#ff4444', linewidth=0.8, alpha=0.5)
    ax3d.plot([0, 0], [-al, al], [0, 0], color='#44ff44', linewidth=0.8, alpha=0.5)
    ax3d.plot([0, 0], [0, 0], [-al, al], color='#4488ff', linewidth=0.8, alpha=0.5)

    # Initial ellipsoid surface
    surface = [ax3d.plot_surface(X0, Y0, Z0, color='#c8a882', alpha=0.75,
                                  linewidth=0, antialiased=True, shade=True)]

    # Sensor placement marker (invisible until placed)
    marker_scatter = [ax3d.scatter([0], [0], [0], color='#00ffff', s=140,
                                    depthshade=False, zorder=10, alpha=0.0)]

    # 2D panel dots
    front_dot, = ax_front.plot([], [], 'o', color='#00ffff', markersize=9,
                                markeredgecolor='#ffffff', markeredgewidth=0.8, zorder=5)
    top_dot,   = ax_top.plot(  [], [], 'o', color='#00ffff', markersize=9,
                                markeredgecolor='#ffffff', markeredgewidth=0.8, zorder=5)

    FLASH_DURATION = 6
    flash = {'frames': 0, 'last_count': 0}

    setup_placement_panels(ax_front, ax_top, HEAD_RADII)

    # ---- Button: Calibrate Axes --------------------------------------------
    # Starts the four-step axis identification wizard
    ax_btn_cal = fig.add_axes([0.04, 0.02, 0.13, 0.05])
    btn_cal = Button(ax_btn_cal, 'Calibrate Axes',
                     color='#223322', hovercolor='#336633')
    btn_cal.label.set_color('white')
    btn_cal.label.set_fontsize(8)

    def on_cal_start(event):
        """Begin the calibration wizard at step 1."""
        global cal_step, cal_baseline_q, cal_captures, axis_map, axis_calibrated
        cal_step        = 1
        cal_baseline_q  = None
        cal_captures    = [None, None, None]
        axis_map        = None
        axis_calibrated = False
        cal_text.set_text(CAL_STEP_LABELS[1])
        fig.canvas.draw_idle()
        print("[CAL] Calibration wizard started.")
        print(f"      {CAL_STEP_LABELS[1]}")

    btn_cal.on_clicked(on_cal_start)

    # ---- Button: Capture ---------------------------------------------------
    # Captures the current quaternion for the active calibration step
    ax_btn_cap = fig.add_axes([0.19, 0.02, 0.10, 0.05])
    btn_cap = Button(ax_btn_cap, 'Capture',
                     color='#332200', hovercolor='#664400')
    btn_cap.label.set_color('white')
    btn_cap.label.set_fontsize(8)

    def on_capture(event):
        """
        Advance the calibration wizard by one step.

        FIX: Added two guards before any quaternion is accepted:
          1. ble_connected check - prevents capturing [1,0,0,0] identity before
             the ESP32 has connected and started sending real IMU data.
          2. Identity quaternion check - catches the edge case where BLE is
             technically connected but the sensor hasn't sent a valid packet yet.
             A real BNO055 in any orientation never returns exact [1,0,0,0].

        Step 1: Capture baseline quaternion (device held still)
        Step 2: Capture pitch-up quaternion
        Step 3: Capture yaw-left quaternion
        Step 4: Capture roll-left quaternion, then compute axis_map
        """
        global cal_step, cal_baseline_q, cal_captures, axis_map, axis_calibrated

        if cal_step == 0:
            print("[CAL] Start calibration first (click 'Calibrate Axes')")
            return

        # Guard 1: BLE must be connected before capturing.
        # Without this, latest_quaternion is the initialization value [1,0,0,0],
        # which makes compute_axis_map produce a garbage (or zero-division) mapping.
        if not ble_connected:
            cal_text.set_text("Not connected to ESP32 - wait for BLE connection")
            fig.canvas.draw_idle()
            print("[CAL] Cannot capture: BLE not connected yet. Wait for connection.")
            return

        # Snapshot the latest remapped quaternion under the lock
        with data_lock:
            q_now = latest_quaternion.copy()

        # Guard 2: Reject identity quaternion [1, 0, 0, 0].
        # The ESP32 initializes latest_quaternion to identity before the first
        # BLE packet arrives. A real BNO055 in NDOF fusion mode will never
        # output exact identity unless the sensor is broken. If we capture this,
        # all delta quaternions in compute_axis_map will be near-zero and the
        # argmax will pick an arbitrary axis.
        if np.allclose(q_now, [1.0, 0.0, 0.0, 0.0], atol=0.01):
            cal_text.set_text("Sensor returning identity quaternion - check BNO055 is calibrated")
            fig.canvas.draw_idle()
            print("[CAL] Identity quaternion detected. BNO055 may not be streaming "
                  "valid fusion data yet. Check calibration status on ESP32 serial monitor.")
            return

        if cal_step == 1:
            # Baseline: device held still
            cal_baseline_q = q_now
            cal_step = 2
            print(f"[CAL] Baseline captured: {q_now.round(4)}")

        elif cal_step == 2:
            # Pitch up
            cal_captures[0] = q_now
            cal_step = 3
            print(f"[CAL] Pitch-up captured: {q_now.round(4)}")

        elif cal_step == 3:
            # Yaw left
            cal_captures[1] = q_now
            cal_step = 4
            print(f"[CAL] Yaw-left captured: {q_now.round(4)}")

        elif cal_step == 4:
            # Roll left - final step, compute the mapping
            cal_captures[2] = q_now
            print(f"[CAL] Roll-left captured: {q_now.round(4)}")

            # compute_axis_map returns [(chip_idx, sign) x3] for [roll, pitch, yaw]
            axis_map        = compute_axis_map(cal_baseline_q,
                                               cal_captures[0],  # pitch
                                               cal_captures[1],  # yaw
                                               cal_captures[2])  # roll
            axis_calibrated = True
            cal_step        = 5
            print(f"[CAL] Axis map: roll={axis_map[0]}, pitch={axis_map[1]}, yaw={axis_map[2]}")
            print("[CAL] Calibration complete. Head model is now remapped.")

        # Update the instruction text for the next step
        if cal_step <= 5:
            cal_text.set_text(CAL_STEP_LABELS[cal_step])
        fig.canvas.draw_idle()

    btn_cap.on_clicked(on_capture)

    # ---- Button: Zero Orientation ------------------------------------------
    ax_btn_zero = fig.add_axes([0.31, 0.02, 0.13, 0.05])
    btn_zero = Button(ax_btn_zero, 'Zero Orientation',
                      color='#1a1a44', hovercolor='#2a2a88')
    btn_zero.label.set_color('white')
    btn_zero.label.set_fontsize(8)

    def on_zero(event):
        """
        Capture current quaternion as the reference pose.
        All future quaternions are expressed relative to this so the current
        physical position becomes the model's neutral (identity) orientation.
        """
        global ref_quaternion
        with data_lock:
            ref_quaternion = latest_quaternion.copy()
        print(f"[ZERO] Reference set: {ref_quaternion.round(4)}")

    btn_zero.on_clicked(on_zero)

    # ---- Button: Place Sensor ----------------------------------------------
    ax_btn_place = fig.add_axes([0.46, 0.02, 0.13, 0.05])
    btn_place = Button(ax_btn_place, 'Place Sensor',
                       color='#1a3322', hovercolor='#2a6644')
    btn_place.label.set_color('white')
    btn_place.label.set_fontsize(8)

    def on_place_btn(event):
        """Activate placement mode - next click on a 2D panel sets the sensor dot."""
        global placement_mode
        placement_mode = True
        place_hint.set_text('PLACEMENT MODE  ->  click Front or Top View')
        fig.canvas.draw_idle()
        print("[PLACE] Click Front or Top View to set sensor position.")

    btn_place.on_clicked(on_place_btn)

    # ---- Button: Recal ESP32 -----------------------------------------------
    ax_btn_recal = fig.add_axes([0.61, 0.02, 0.13, 0.05])
    btn_recal = Button(ax_btn_recal, 'Recal ESP32',
                       color='#331a1a', hovercolor='#883322')
    btn_recal.label.set_color('white')
    btn_recal.label.set_fontsize(8)

    def on_recal_esp32(event):
        """
        Queue a RECAL command to be sent via the existing BLE streaming connection.
        Also resets the software zero reference back to identity.

        FIX: Previously called send_command_async which spawned a new thread
        opening a second BleakClient. That second connection dropped the main
        IMU streaming connection because the ESP32 supports only one connection
        at a time. Now uses the pending_ble_command flag so the existing
        ble_receiver connection handles the write on its next tick.
        """
        global ref_quaternion
        ref_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        send_command_async("RECAL")
        print("[RECAL] RECAL queued. ESP32 will re-init BNO055 on next loop.")
        print("[RECAL] Re-perform calibration routine after sensor restarts.")

    btn_recal.on_clicked(on_recal_esp32)

    # ---- Placement click handler -------------------------------------------
    def on_placement_click(event):
        """
        Convert a 2D click in Front or Top panel to spherical sensor placement angles.

        Front View (XZ plane):
            sensor_az = arcsin(click_x / rx)   left-right angle
            sensor_el = arcsin(click_z / rz)   up-down angle

        Top View (XY plane):
            sensor_az = atan2(click_x / rx, click_y / ry)   full 360 azimuth
            sensor_el unchanged
        """
        global sensor_az, sensor_el, sensor_placed, placement_mode

        if not placement_mode:
            return
        if event.xdata is None or event.ydata is None:
            return

        rx, ry, rz = HEAD_RADII

        if event.inaxes == ax_front:
            u = float(np.clip(event.xdata / rx, -1.0, 1.0))
            v = float(np.clip(event.ydata / rz, -1.0, 1.0))
            sensor_az      = np.arcsin(u)
            sensor_el      = np.arcsin(v)
            sensor_placed  = True
            placement_mode = False
            place_hint.set_text('')
            print(f"[PLACE] Front view -> az={np.degrees(sensor_az):.1f} deg, "
                  f"el={np.degrees(sensor_el):.1f} deg")

        elif event.inaxes == ax_top:
            u = float(np.clip(event.xdata / rx, -1.0, 1.0))
            v = float(np.clip(event.ydata / ry, -1.0, 1.0))
            sensor_az      = float(np.arctan2(u, v))
            sensor_placed  = True
            placement_mode = False
            place_hint.set_text('')
            print(f"[PLACE] Top view -> az={np.degrees(sensor_az):.1f} deg, "
                  f"el={np.degrees(sensor_el):.1f} deg (elevation unchanged)")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_placement_click)

    # ---- Animation update function -----------------------------------------
    def update(frame):
        # Snapshot all shared state under a single lock acquisition
        with data_lock:
            q_raw     = latest_quaternion.copy()
            accel     = latest_linear_accel.copy()
            ts        = latest_timestamp_ms
            n_impacts = len(impact_history)
            connected = ble_connected
            q_ref     = ref_quaternion.copy()
            s_az      = sensor_az
            s_el      = sensor_el
            placed    = sensor_placed

        # Apply software zero: remove the reference pose offset.
        # If no zero set, q_ref is identity and this is a no-op.
        q = apply_reference_quaternion(q_raw, q_ref)

        R = quaternion_to_rotation_matrix(q)
        Xr, Yr, Zr = rotate_ellipsoid(X0, Y0, Z0, R)

        surface[0].remove()

        if n_impacts > flash['last_count']:
            flash['frames']     = FLASH_DURATION
            flash['last_count'] = n_impacts

        surf_color = '#ff3333' if flash['frames'] > 0 else '#c8a882'
        if flash['frames'] > 0:
            flash['frames'] -= 1

        surface[0] = ax3d.plot_surface(Xr, Yr, Zr, color=surf_color, alpha=0.75,
                                        linewidth=0, antialiased=True, shade=True)

        # Sensor marker: base position rotated with the head
        marker_scatter[0].remove()
        if placed:
            m_base = sensor_position_3d(s_az, s_el, HEAD_RADII)
            m_rot  = R @ m_base
            marker_scatter[0] = ax3d.scatter(
                [m_rot[0]], [m_rot[1]], [m_rot[2]],
                color='#00ffff', s=140, depthshade=False, zorder=10, alpha=1.0
            )
            front_dot.set_data([np.sin(s_az) * HEAD_RADII[0]],
                                [np.sin(s_el) * HEAD_RADII[2]])
            top_dot.set_data([np.cos(s_el) * np.sin(s_az) * HEAD_RADII[0]],
                              [np.cos(s_el) * np.cos(s_az) * HEAD_RADII[1]])
        else:
            marker_scatter[0] = ax3d.scatter(
                [0], [0], [0], color='#00ffff', s=1,
                depthshade=False, zorder=10, alpha=0.0
            )
            front_dot.set_data([], [])
            top_dot.set_data([], [])

        # Status overlay
        if connected:
            mag = np.linalg.norm(accel)
            cal_status = "calibrated" if axis_calibrated else "NOT calibrated - click Calibrate Axes"
            status_text.set_text(
                f"Connected  |  ||a|| = {mag:.2f} m/s^2 ({mag/9.81:.2f}g)"
                f"  |  t = {ts} ms  |  axes: {cal_status}"
            )
            status_text.set_color('#44ff88')
        else:
            status_text.set_text("Connecting to ESP32... (auto-retry active)")
            status_text.set_color('#ffcc44')

        impact_text.set_text(
            f"Impacts: {n_impacts}"
            f"  |  Player: {player_name}  (ESP32: {esp32_player_id})"
        )

        return surface[0],

    def on_close(event):
        global running
        print("[RENDER] Window closed.")
        running = False

    fig.canvas.mpl_connect('close_event', on_close)

    anim = animation.FuncAnimation(fig, update, interval=33,
                                   blit=False, cache_frame_data=False)

    print("[RENDER] 3D window opened.")
    print("  Recommended workflow:")
    print("    1. Wait for 'Connected' status before clicking any buttons")
    print("    2. Click 'Calibrate Axes' and follow the four steps")
    print("    3. Click 'Zero Orientation' to set neutral pose")
    print("    4. Click 'Place Sensor' then click Front/Top view")
    print("    5. Click 'Recal ESP32' only if the BNO055 needs a full hardware reset\n")

    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    print("=" * 65)
    print("  Head Impact Monitor - Python Receiver")
    print(f"  Impact threshold: {IMPACT_THRESHOLD_MS2} m/s^2 "
          f"({IMPACT_THRESHOLD_MS2/9.81:.2f}g)")
    print("=" * 65 + "\n")

    # Player name
    player_name = input("Enter player name (press Enter for 'Player1'): ").strip()
    if not player_name:
        player_name = "Player1"
    print(f"  Session player: {player_name}\n")

    # Head dimensions
    print("Head dimensions (defaults):")
    print(f"  Width  (X): {HEAD_WIDTH_CM} cm")
    print(f"  Depth  (Y): {HEAD_DEPTH_CM} cm")
    print(f"  Height (Z): {HEAD_HEIGHT_CM} cm")
    resp = input("\nPress Enter to use these, or type 'edit' to change: ").strip().lower()

    if resp == "edit":
        try:
            HEAD_WIDTH_CM  = float(input("  Width  (cm): "))
            HEAD_DEPTH_CM  = float(input("  Depth  (cm): "))
            HEAD_HEIGHT_CM = float(input("  Height (cm): "))
            HEAD_RADII[:] = [HEAD_WIDTH_CM/2, HEAD_DEPTH_CM/2, HEAD_HEIGHT_CM/2]
            print(f"  Updated: {HEAD_WIDTH_CM} x {HEAD_DEPTH_CM} x {HEAD_HEIGHT_CM} cm\n")
        except ValueError:
            print("[WARN] Invalid input. Using defaults.\n")

    # Start BLE in background daemon thread
    ble_thread = threading.Thread(target=ble_thread_func, daemon=True)
    ble_thread.start()

    time.sleep(1.0)  # Let BLE scan output print before window opens

    try:
        run_render_loop()
    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt. Shutting down...")
        running = False

    ble_thread.join(timeout=3.0)
    save_impact_log(player_name, esp32_player_id, sensor_az, sensor_el)
    print("[MAIN] Session ended.")
