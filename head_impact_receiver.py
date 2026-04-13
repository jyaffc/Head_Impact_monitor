# =============================================================================
# head_impact_receiver.py
# Python BLE Receiver - Head Impact Monitor
#
# Purpose:
#   Receives BLE data packets from the ESP32 HeadImpactMonitor device,
#   renders a real-time 3D head model (ellipsoid) that rotates to match the
#   user's head orientation using quaternion data, detects head impacts based
#   on linear acceleration magnitude, and logs impact history to a CSV file.
#
# Required libraries (install via pip before running):
#   pip install bleak numpy matplotlib
#
#   All three work on Python 3.13. No version conflicts.
#
# Usage:
#   1. Power on the ESP32 with the HeadImpactMonitor sketch loaded
#   2. Perform the BNO055 calibration routine (figure-8 movements)
#   3. Run this script: python head_impact_receiver.py
#   4. The script will scan for the ESP32, connect, and open the 3D viewer
#   5. Close the matplotlib window or press Ctrl+C to stop and save the log
#
# Output:
#   - Real-time matplotlib window showing the rotating 3D head ellipsoid
#   - Console alerts when impacts above threshold are detected
#   - CSV file: impact_log_YYYY-MM-DD_HH-MM-SS.csv saved in the same folder
# =============================================================================

import asyncio
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D          # Enables 3D projection in matplotlib
import csv
import datetime
import threading
import time
from bleak import BleakClient, BleakScanner

# =============================================================================
# CONFIGURATION - Must match values in HeadImpactMonitor.ino exactly
# =============================================================================

BLE_DEVICE_NAME = "HeadImpactMonitor"
SERVICE_UUID    = "12345678-1234-1234-1234-123456789abc"
CHAR_UUID_IMU   = "abcd1234-ab12-ab12-ab12-abcdef123456"

# '<' = little-endian, '10f' = ten 32-bit floats, 'I' = one uint32
PACKET_FORMAT = "<10fI"
PACKET_SIZE   = struct.calcsize(PACKET_FORMAT)  # 44 bytes

# Impact detection threshold in m/s^2 (gravity already removed by BNO055)
# 14.7 m/s^2 = 1.5g of net dynamic force
IMPACT_THRESHOLD_MS2 = 14.7

# =============================================================================
# HEAD MODEL DIMENSIONS
# Measure the user's head and enter values in centimeters.
# X = left-right width, Y = front-back depth, Z = top-bottom height
# Average adult: ~15cm wide, ~19cm deep, ~23cm tall
# =============================================================================

HEAD_WIDTH_CM  = 15.0
HEAD_DEPTH_CM  = 19.0
HEAD_HEIGHT_CM = 23.0

# Half-dimensions (radii) used for the ellipsoid parameterization
HEAD_RADII = np.array([
    HEAD_WIDTH_CM  / 2.0,
    HEAD_DEPTH_CM  / 2.0,
    HEAD_HEIGHT_CM / 2.0
])

# =============================================================================
# SHARED STATE - written by BLE thread, read by render thread
# =============================================================================

latest_quaternion   = np.array([1.0, 0.0, 0.0, 0.0])  # Identity = no rotation
latest_linear_accel = np.array([0.0, 0.0, 0.0])
latest_timestamp_ms = 0
impact_history      = []
data_lock           = threading.Lock()
running             = True
ble_connected       = False


# =============================================================================
# QUATERNION TO ROTATION MATRIX
# Converts a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.
# The rotation matrix is applied to ellipsoid vertices each animation frame
# to rotate the 3D model to match the IMU's measured orientation.
# =============================================================================
def quaternion_to_rotation_matrix(q):
    """
    Convert unit quaternion to 3x3 rotation matrix.

    Parameters:
        q (np.ndarray): [w, x, y, z]

    Returns:
        np.ndarray: 3x3 rotation matrix R such that v_rotated = R @ v
    """
    w, x, y, z = q
    xx = x*x; yy = y*y; zz = z*z
    xy = x*y; xz = x*z; yz = y*z
    wx = w*x; wy = w*y; wz = w*z

    return np.array([
        [1 - 2*(yy+zz),   2*(xy-wz),   2*(xz+wy)  ],
        [  2*(xy+wz),   1 - 2*(xx+zz), 2*(yz-wx)   ],
        [  2*(xz-wy),     2*(yz+wx),   1 - 2*(xx+yy)]
    ])


# =============================================================================
# BUILD ELLIPSOID VERTICES
# Generates XYZ vertex grids for an ellipsoid using spherical parameterization.
#
# Spherical parameterization:
#   x = rx * sin(phi) * cos(theta)
#   y = ry * sin(phi) * sin(theta)
#   z = rz * cos(phi)
# where phi in [0, pi] is the polar angle, theta in [0, 2*pi] is azimuthal.
#
# Returns 2D grids (meshgrids) compatible with matplotlib's plot_surface().
# =============================================================================
def build_ellipsoid(radii, resolution=24):
    """
    Generate ellipsoid surface vertex grids.

    Parameters:
        radii (np.ndarray): [rx, ry, rz] semi-axes in cm
        resolution (int): angular grid resolution (higher = smoother but slower)

    Returns:
        tuple: (X, Y, Z) each a (resolution x resolution) np.ndarray
    """
    phi   = np.linspace(0, np.pi,    resolution)
    theta = np.linspace(0, 2*np.pi,  resolution)
    phi, theta = np.meshgrid(phi, theta)

    X = radii[0] * np.sin(phi) * np.cos(theta)
    Y = radii[1] * np.sin(phi) * np.sin(theta)
    Z = radii[2] * np.cos(phi)

    return X, Y, Z


# =============================================================================
# ROTATE ELLIPSOID VERTICES
# Applies a 3x3 rotation matrix R to all ellipsoid surface vertices at once.
# Uses numpy matrix multiplication to rotate all points simultaneously,
# which is faster than looping over individual vertices.
# =============================================================================
def rotate_ellipsoid(X, Y, Z, R):
    """
    Apply rotation matrix R to ellipsoid vertex grids.

    Parameters:
        X, Y, Z (np.ndarray): vertex grids from build_ellipsoid()
        R (np.ndarray): 3x3 rotation matrix

    Returns:
        tuple: (Xr, Yr, Zr) rotated vertex grids in same grid shape
    """
    shape = X.shape
    # Flatten grids into column vectors and stack into (3, N) matrix
    verts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    # Rotate all N vertices simultaneously: (3,3) @ (3,N) = (3,N)
    verts_rotated = R @ verts
    # Reshape back to 2D grid for plot_surface()
    return (verts_rotated[0].reshape(shape),
            verts_rotated[1].reshape(shape),
            verts_rotated[2].reshape(shape))


# =============================================================================
# IMPACT LOGGING
# =============================================================================
def save_impact_log():
    """Save all recorded impact events to a timestamped CSV file."""
    if not impact_history:
        print("[LOG] No impacts recorded this session.")
        return

    filename = datetime.datetime.now().strftime("impact_log_%Y-%m-%d_%H-%M-%S.csv")

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "datetime",
            "esp32_timestamp_ms",
            "accel_magnitude_ms2",
            "accel_magnitude_g",
            "direction_x",
            "direction_y",
            "direction_z"
        ])
        for event in impact_history:
            writer.writerow([
                event["datetime"],
                event["timestamp_ms"],
                f"{event['magnitude_ms2']:.3f}",
                f"{event['magnitude_g']:.3f}",
                f"{event['direction'][0]:.4f}",
                f"{event['direction'][1]:.4f}",
                f"{event['direction'][2]:.4f}"
            ])

    print(f"[LOG] Saved: {filename}  ({len(impact_history)} events)")


# =============================================================================
# BLE NOTIFICATION CALLBACK
# Called automatically by bleak on every incoming BLE notify packet (~100 Hz).
# Unpacks the 44-byte binary packet, checks for impacts, and writes to shared
# state so the render thread can read the latest orientation.
# =============================================================================
def on_imu_notification(sender, raw_data):
    """
    BLE notification handler. Called on every incoming IMU packet.

    Parameters:
        sender: BLE characteristic handle (required by bleak API, unused here)
        raw_data (bytearray): 44-byte binary packet from ESP32
    """
    global latest_quaternion, latest_linear_accel, latest_timestamp_ms
    global impact_history

    if len(raw_data) != PACKET_SIZE:
        print(f"[WARN] Bad packet size: {len(raw_data)} bytes (expected {PACKET_SIZE})")
        return

    # Unpack little-endian binary: 10 float32 values then 1 uint32
    (qw, qx, qy, qz,
     linAccX, linAccY, linAccZ,
     gyroX, gyroY, gyroZ,
     timestamp_ms) = struct.unpack(PACKET_FORMAT, raw_data)

    # Compute net linear acceleration magnitude
    accel_vec       = np.array([linAccX, linAccY, linAccZ])
    accel_magnitude = np.linalg.norm(accel_vec)

    # Impact detection: threshold check on net acceleration magnitude
    if accel_magnitude > IMPACT_THRESHOLD_MS2:
        direction = accel_vec / accel_magnitude if accel_magnitude > 0 else accel_vec
        event = {
            "datetime":      datetime.datetime.now().isoformat(),
            "timestamp_ms":  timestamp_ms,
            "magnitude_ms2": accel_magnitude,
            "magnitude_g":   accel_magnitude / 9.81,
            "direction":     direction
        }
        with data_lock:
            impact_history.append(event)

        print(f"[IMPACT] {event['datetime']} | "
              f"{accel_magnitude:.2f} m/s^2 ({accel_magnitude/9.81:.2f}g) | "
              f"dir=[{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")

    # Write latest sensor values to shared state (protected by lock)
    with data_lock:
        latest_quaternion   = np.array([qw, qx, qy, qz])
        latest_linear_accel = accel_vec
        latest_timestamp_ms = timestamp_ms


# =============================================================================
# BLE RECEIVER (async, runs in background thread)
# =============================================================================
async def ble_receiver():
    """Scan for ESP32, connect, and stream IMU data until running=False."""
    global running, ble_connected

    print(f"[BLE] Scanning for '{BLE_DEVICE_NAME}'...")

    device = await BleakScanner.find_device_by_name(BLE_DEVICE_NAME, timeout=15.0)

    if device is None:
        print(f"[ERROR] '{BLE_DEVICE_NAME}' not found within 15 seconds.")
        print("  -> Is the ESP32 powered on and advertising?")
        running = False
        return

    print(f"[BLE] Found: {device.name} ({device.address})")

    async with BleakClient(device) as client:
        print(f"[BLE] Connected. MTU: {client.mtu_size} bytes")
        await client.start_notify(CHAR_UUID_IMU, on_imu_notification)
        ble_connected = True
        print(f"[BLE] Streaming at ~100 Hz | threshold: "
              f"{IMPACT_THRESHOLD_MS2} m/s^2 ({IMPACT_THRESHOLD_MS2/9.81:.2f}g)\n")

        while running:
            await asyncio.sleep(0.1)

        await client.stop_notify(CHAR_UUID_IMU)
        print("[BLE] Disconnected.")


def ble_thread_func():
    """Entry point for the background BLE thread."""
    asyncio.run(ble_receiver())


# =============================================================================
# MATPLOTLIB 3D RENDER LOOP (main thread)
# FuncAnimation calls update() every 33ms (~30 Hz).
# Each frame reads the latest quaternion, rotates the ellipsoid vertices,
# removes the old surface, and draws a new one with the updated orientation.
# Impact events trigger a red color flash on the ellipsoid for 6 frames.
# =============================================================================
def run_render_loop():
    """Build the matplotlib figure and run the animation render loop."""
    global running

    # Build base (unrotated) ellipsoid vertex grids
    X0, Y0, Z0 = build_ellipsoid(HEAD_RADII, resolution=24)

    # --- Figure setup ---
    fig = plt.figure(figsize=(9, 7))
    fig.patch.set_facecolor('#1a1a2e')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')

    max_dim = float(max(HEAD_RADII)) * 1.4
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.set_zlim(-max_dim, max_dim)

    ax.set_xlabel('X  (left-right)', color='#cccccc', labelpad=8)
    ax.set_ylabel('Y  (front-back)', color='#cccccc', labelpad=8)
    ax.set_zlabel('Z  (up-down)',    color='#cccccc', labelpad=8)
    ax.tick_params(colors='#888888')
    ax.set_title('Head Impact Monitor — Live 3D Orientation',
                 color='white', fontsize=12, pad=14)

    # Status and impact overlays in figure coordinates (not 3D axes)
    status_text = fig.text(0.02, 0.96, 'Connecting to ESP32...',
                           color='#ffcc44', fontsize=9,
                           verticalalignment='top', fontfamily='monospace')

    impact_text = fig.text(0.98, 0.96, 'Impacts: 0',
                           color='#ff6666', fontsize=9,
                           verticalalignment='top', horizontalalignment='right',
                           fontfamily='monospace')

    # Fixed world-space reference axes (do not rotate with the head)
    axis_len = max_dim * 0.9
    ax.plot([-axis_len, axis_len], [0, 0], [0, 0],
            color='#ff4444', linewidth=0.8, alpha=0.5)  # X - red
    ax.plot([0, 0], [-axis_len, axis_len], [0, 0],
            color='#44ff44', linewidth=0.8, alpha=0.5)  # Y - green
    ax.plot([0, 0], [0, 0], [-axis_len, axis_len],
            color='#4488ff', linewidth=0.8, alpha=0.5)  # Z - blue

    # Initial surface at identity orientation
    surface = [ax.plot_surface(X0, Y0, Z0,
                               color='#c8a882', alpha=0.75,
                               linewidth=0, antialiased=True, shade=True)]

    # Impact flash state tracker
    FLASH_DURATION = 6   # frames to show red color after an impact
    flash = {'frames': 0, 'last_count': 0}

    # --------------------------------------------------------------------------
    # Animation update function - called every 33ms by FuncAnimation
    # --------------------------------------------------------------------------
    def update(frame):
        with data_lock:
            q         = latest_quaternion.copy()
            accel     = latest_linear_accel.copy()
            ts        = latest_timestamp_ms
            n_impacts = len(impact_history)
            connected = ble_connected

        # Rotate ellipsoid vertices by the latest quaternion
        R = quaternion_to_rotation_matrix(q)
        Xr, Yr, Zr = rotate_ellipsoid(X0, Y0, Z0, R)

        # Remove old surface and redraw (plot_surface does not support vertex updates)
        surface[0].remove()

        # Impact flash: switch to red for FLASH_DURATION frames after each new impact
        if n_impacts > flash['last_count']:
            flash['frames']     = FLASH_DURATION
            flash['last_count'] = n_impacts

        surf_color = '#ff3333' if flash['frames'] > 0 else '#c8a882'
        if flash['frames'] > 0:
            flash['frames'] -= 1

        surface[0] = ax.plot_surface(Xr, Yr, Zr,
                                     color=surf_color, alpha=0.75,
                                     linewidth=0, antialiased=True, shade=True)

        # Update overlays
        if connected:
            mag = np.linalg.norm(accel)
            status_text.set_text(
                f"Connected  |  ||a|| = {mag:.2f} m/s^2 ({mag/9.81:.2f}g)"
                f"  |  t = {ts} ms"
            )
            status_text.set_color('#44ff88')
        else:
            status_text.set_text("Connecting to ESP32...")
            status_text.set_color('#ffcc44')

        impact_text.set_text(f"Impacts: {n_impacts}")
        return surface[0],

    # Stop running flag when window is closed
    def on_close(event):
        global running
        print("[RENDER] Window closed.")
        running = False

    fig.canvas.mpl_connect('close_event', on_close)

    # FuncAnimation: calls update() every 33ms (~30 Hz)
    # blit=False is required for 3D axes (blit only works in 2D matplotlib)
    anim = animation.FuncAnimation(fig, update,
                                   interval=33, blit=False,
                                   cache_frame_data=False)

    print("[RENDER] 3D window opened.")
    print("         Move the ESP32 to see the head model rotate.")
    print("         Close the window or press Ctrl+C to stop.\n")

    plt.show()  # Blocks until window is closed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    print("=" * 62)
    print("  Head Impact Monitor - Python Receiver")
    print(f"  Head model: {HEAD_WIDTH_CM}cm W x {HEAD_DEPTH_CM}cm D x {HEAD_HEIGHT_CM}cm H")
    print(f"  Impact threshold: {IMPACT_THRESHOLD_MS2} m/s^2"
          f" ({IMPACT_THRESHOLD_MS2/9.81:.2f}g)")
    print("=" * 62 + "\n")

    print("Head dimensions loaded from config:")
    print(f"  Width  (X, left-right): {HEAD_WIDTH_CM} cm")
    print(f"  Depth  (Y, front-back): {HEAD_DEPTH_CM} cm")
    print(f"  Height (Z, up-down):    {HEAD_HEIGHT_CM} cm")
    response = input(
        "\nPress Enter to use these, or type 'edit' to change them: "
    ).strip().lower()

    if response == "edit":
        try:
            HEAD_WIDTH_CM  = float(input("  Head width  (cm, left-right): "))
            HEAD_DEPTH_CM  = float(input("  Head depth  (cm, front-back): "))
            HEAD_HEIGHT_CM = float(input("  Head height (cm, up-down):    "))
            HEAD_RADII[:] = [
                HEAD_WIDTH_CM  / 2.0,
                HEAD_DEPTH_CM  / 2.0,
                HEAD_HEIGHT_CM / 2.0
            ]
            print(f"  Updated: {HEAD_WIDTH_CM} x {HEAD_DEPTH_CM} x {HEAD_HEIGHT_CM} cm\n")
        except ValueError:
            print("[WARN] Invalid input. Using defaults.\n")

    # Start BLE in a background daemon thread
    ble_thread = threading.Thread(target=ble_thread_func, daemon=True)
    ble_thread.start()

    time.sleep(1.0)  # Let BLE scanning output print before window opens

    try:
        run_render_loop()  # Must run on main thread (matplotlib requirement)
    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt. Shutting down...")
        running = False

    ble_thread.join(timeout=3.0)
    save_impact_log()
    print("[MAIN] Session ended.")