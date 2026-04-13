// =============================================================================
// HeadImpactMonitor.ino
// ESP32-SOLO-1 + BNO055 IMU - Head Orientation & Impact Data Streamer
// WIRING (breadboard, matches wiring diagram):
//   BNO055 VIN  ->  ESP32 3V3       (power - must be 3.3V not 5V)
//   BNO055 GND  ->  ESP32 GND       (ground)
//   BNO055 SDA  ->  ESP32 GPIO 21   (I2C data)
//   BNO055 SCL  ->  ESP32 GPIO 22   (I2C clock)
//   BNO055 ADR  ->  ESP32 GND       (sets I2C address to 0x28)
//
// HOW TO UPLOAD:
//   1. Connect ESP32 to PC via USB
//   2. Select the correct COM port in Tools > Port
//   3. Hold the BOOT button on the ESP32, click Upload in Arduino IDE,
//      release BOOT once "Connecting..." appears in the console
//      (some boards upload without needing this - try without first)
//   4. Once upload completes, press the EN (reset) button on the ESP32
//   5. Open Serial Monitor at 115200 baud to watch calibration output
//
// FIRST RUN - CALIBRATION:
//   The BNO055 must be calibrated before orientation data is reliable.
//   Watch the Serial Monitor for calibration status: Sys/Gyro/Accel/Mag
//   Each number goes from 0 (uncalibrated) to 3 (fully calibrated).
//   - For gyroscope: set the device flat and completely still for ~2 seconds
//   - For accelerometer: hold in 6 different orientations (all faces)
//   - For magnetometer: move in a figure-8 pattern slowly in the air
//   - System reaches 3 once all subsystems are sufficiently calibrated
//   Minimum to stream reliable data: Sys >= 1, Gyro >= 1
//
// BLE DATA FORMAT:
//   Each BLE notification is a 44-byte binary packet (IMUPacket struct).
//   The Python receiver unpacks this with struct.unpack("<10fI", data).
//   Packet layout:
//     bytes  0-15 : qw, qx, qy, qz         (float32 x4) quaternion
//     bytes 16-27 : linAccX, linAccY, linAccZ (float32 x3) m/s^2
//     bytes 28-39 : gyroX, gyroY, gyroZ    (float32 x3) deg/s
//     bytes 40-43 : timestamp              (uint32) ms since boot
// =============================================================================

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// =============================================================================
// CONFIGURATION
// Only change these if your hardware setup differs from the wiring diagram
// =============================================================================

// BNO055 I2C address
// 0x28 -> ADR pin tied to GND (our wiring)
// 0x29 -> ADR pin tied to 3V3
#define BNO055_I2C_ADDR  0x28

// Sampling interval in milliseconds
// 10ms = 100Hz which is the maximum the BNO055 fusion output supports
#define SAMPLE_INTERVAL_MS  10

// BLE device name - the Python script scans for exactly this string
#define BLE_DEVICE_NAME  "HeadImpactMonitor"

// BLE service and characteristic UUIDs
// These must match exactly in the Python receiver script
#define SERVICE_UUID    "12345678-1234-1234-1234-123456789abc"
#define CHAR_UUID_IMU   "abcd1234-ab12-ab12-ab12-abcdef123456"

// Local impact pre-alert threshold in m/s^2 for serial monitor printout only
// The BNO055 removes gravity so this is net dynamic acceleration
// 9.81 m/s^2 = 1g, so 14.7 = 1.5g
// Tune this value after testing once hardware is assembled
#define IMPACT_ACCEL_THRESHOLD_MS2  14.7f


// =============================================================================
// IMU DATA PACKET STRUCTURE
//
// This struct defines the exact 44-byte binary layout of each BLE transmission.
// #pragma pack(push, 1) tells the compiler to use no padding bytes between
// struct members - without this the compiler may insert invisible padding bytes
// that would shift the byte positions and corrupt the Python unpacking.
//
// Python unpacks with: struct.unpack("<10fI", raw_data)
//   "<"  = little-endian byte order (ESP32 is little-endian)
//   "10f" = ten 32-bit floats
//   "I"  = one unsigned 32-bit integer
// =============================================================================
#pragma pack(push, 1)
struct IMUPacket {
    float    qw;          // Quaternion scalar (w) component
    float    qx;          // Quaternion vector (x) component
    float    qy;          // Quaternion vector (y) component
    float    qz;          // Quaternion vector (z) component
    float    linAccX;     // Linear acceleration X in m/s^2 (gravity removed)
    float    linAccY;     // Linear acceleration Y in m/s^2
    float    linAccZ;     // Linear acceleration Z in m/s^2
    float    gyroX;       // Angular velocity X in deg/s
    float    gyroY;       // Angular velocity Y in deg/s
    float    gyroZ;       // Angular velocity Z in deg/s
    uint32_t timestamp;   // Time in milliseconds since device power-on
};
#pragma pack(pop)


// =============================================================================
// GLOBAL OBJECTS AND STATE
// =============================================================================

// BNO055 sensor object
// First argument: arbitrary unique sensor ID used internally by Adafruit library
// Second argument: I2C address
Adafruit_BNO055 bno = Adafruit_BNO055(55, BNO055_I2C_ADDR);

// BLE server and characteristic - assigned during setup(), used in loop()
BLEServer*         pServer            = nullptr;
BLECharacteristic* pIMUCharacteristic = nullptr;

// BLE connection state flags
// deviceConnected:    true while a Python client is actively connected
// oldDeviceConnected: previous frame's connection state, used to detect changes
bool deviceConnected    = false;
bool oldDeviceConnected = false;

// Calibration status variables (0-3 scale, 3 = fully calibrated)
uint8_t calSys, calGyro, calAccel, calMag;


// =============================================================================
// BLE SERVER CALLBACKS
// These methods are called automatically by the ESP32 BLE stack when the
// Python client connects or disconnects. No manual polling needed.
// =============================================================================
class ServerCallbacks : public BLEServerCallbacks {

    // Triggered when Python client connects via BLE
    void onConnect(BLEServer* pServer) override {
        deviceConnected = true;
        Serial.println("[BLE] Client connected - streaming started");
    }

    // Triggered when Python client disconnects or connection is lost
    void onDisconnect(BLEServer* pServer) override {
        deviceConnected = false;
        Serial.println("[BLE] Client disconnected - restarting advertising");
    }
};


// =============================================================================
// SETUP - Runs once on power-on or after pressing the EN (reset) button
// =============================================================================
void setup() {

    // Start serial communication for debug output
    // Open Serial Monitor in Arduino IDE at this baud rate to see output
    Serial.begin(115200);
    delay(500);
    Serial.println();
    Serial.println("==============================================");
    Serial.println("  HeadImpactMonitor - ESP32-SOLO-1 + BNO055");
    Serial.println("==============================================");

    // -------------------------------------------------------------------------
    // Initialize I2C bus
    // ESP32 default I2C pins: SDA = GPIO 21, SCL = GPIO 22
    // These match our wiring diagram - no custom pin config needed
    // -------------------------------------------------------------------------
    Wire.begin();
    Serial.println("[I2C] Bus initialized on GPIO 21 (SDA), GPIO 22 (SCL)");

    // -------------------------------------------------------------------------
    // Initialize BNO055 sensor
    // bno.begin() returns false if the sensor is not detected on I2C
    // If this fails, check: wiring, VIN=3.3V, ADR pin, I2C address (0x28/0x29)
    // -------------------------------------------------------------------------
    Serial.print("[IMU] Initializing BNO055 at I2C address 0x");
    Serial.println(BNO055_I2C_ADDR, HEX);

    if (!bno.begin()) {
        Serial.println("[ERROR] BNO055 not detected. Check wiring:");
        Serial.println("        VIN -> 3V3, GND -> GND");
        Serial.println("        SDA -> GPIO 21, SCL -> GPIO 22");
        Serial.println("        ADR -> GND (for address 0x28)");
        Serial.println("  Halting. Fix wiring and press EN to retry.");
        while (true) {
            delay(1000); // Halt here - do not proceed without IMU
        }
    }

    Serial.println("[IMU] BNO055 detected successfully");

    // Use the external 32.768 kHz crystal on the Adafruit breakout board
    // for better timing accuracy in the internal sensor fusion algorithm
    bno.setExtCrystalUse(true);
    Serial.println("[IMU] External crystal enabled");

    // The BNO055 defaults to NDOF (Nine Degrees Of Freedom) fusion mode
    // In NDOF mode the chip internally fuses accelerometer + gyroscope +
    // magnetometer to output a calibrated absolute orientation quaternion.
    // This runs entirely on the BNO055 chip - the ESP32 just reads the result.
    Serial.println("[IMU] Fusion mode: NDOF (9-DOF absolute orientation)");
    Serial.println("[IMU] Sampling rate: 100 Hz (10ms interval)");

    // -------------------------------------------------------------------------
    // Initialize BLE
    // -------------------------------------------------------------------------
    BLEDevice::init(BLE_DEVICE_NAME);
    Serial.print("[BLE] Device initialized as: ");
    Serial.println(BLE_DEVICE_NAME);

    // Create BLE server and attach connection/disconnection callbacks
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    // Create a BLE service - a logical container for related characteristics
    BLEService* pService = pServer->createService(SERVICE_UUID);

    // Create the IMU data characteristic inside the service
    // PROPERTY_NOTIFY: server pushes data to client automatically without
    // the client needing to request/poll each value
    pIMUCharacteristic = pService->createCharacteristic(
        CHAR_UUID_IMU,
        BLECharacteristic::PROPERTY_NOTIFY
    );

    // BLE2902 is the standard Client Characteristic Configuration Descriptor
    // Required for NOTIFY to function - allows the Python client to subscribe
    pIMUCharacteristic->addDescriptor(new BLE2902());

    // Start the service (makes the characteristic visible to connecting clients)
    pService->start();

    // Begin advertising so the Python script can discover the device
    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    BLEDevice::startAdvertising();

    Serial.println("[BLE] Advertising started - Python script can now connect");
    Serial.println();
    Serial.println("----------------------------------------------");
    Serial.println("  CALIBRATION - move device to calibrate:");
    Serial.println("  Gyro:  hold completely still for 2 seconds");
    Serial.println("  Accel: tilt to 6 different orientations");
    Serial.println("  Mag:   move in slow figure-8 pattern");
    Serial.println("  Target: Sys=3 Gyro=3 Accel=3 Mag=3");
    Serial.println("  Minimum to stream: Sys>=1, Gyro>=1");
    Serial.println("----------------------------------------------");
    Serial.println();
}


// =============================================================================
// LOOP - Runs continuously after setup() completes
// =============================================================================
void loop() {

    // Timestamps for controlling sample rate and calibration print rate
    static uint32_t lastSampleTime  = 0;
    static uint32_t lastCalPrintTime = 0;

    uint32_t now = millis(); // Current time in ms since power-on

    // -------------------------------------------------------------------------
    // BLE Reconnection Handler
    // Detects when a client disconnects and restarts advertising automatically
    // so the Python script can reconnect without rebooting the ESP32
    // -------------------------------------------------------------------------
    if (!deviceConnected && oldDeviceConnected) {
        // Client just disconnected - wait briefly for BLE stack to clean up
        delay(500);
        pServer->startAdvertising();
        oldDeviceConnected = false;
        Serial.println("[BLE] Advertising restarted - waiting for reconnection");
    }

    // Track when a new connection is established
    if (deviceConnected && !oldDeviceConnected) {
        oldDeviceConnected = true;
        Serial.println("[BLE] New connection established");
    }

    // -------------------------------------------------------------------------
    // Calibration Status Printout (when not streaming)
    // Prints every 2 seconds so you can monitor calibration progress
    // in the Arduino IDE Serial Monitor before connecting the Python script
    // -------------------------------------------------------------------------
    if (!deviceConnected && (now - lastCalPrintTime >= 2000)) {
        lastCalPrintTime = now;

        // getCalibration fills each variable with a value 0-3
        bno.getCalibration(&calSys, &calGyro, &calAccel, &calMag);

        Serial.print("[CAL] Sys:");
        Serial.print(calSys);
        Serial.print("  Gyro:");
        Serial.print(calGyro);
        Serial.print("  Accel:");
        Serial.print(calAccel);
        Serial.print("  Mag:");
        Serial.println(calMag);

        if (calSys >= 1 && calGyro >= 1) {
            Serial.println("      Ready - run Python script to connect");
        } else {
            Serial.println("      Still calibrating...");
        }
    }

    // -------------------------------------------------------------------------
    // IMU Sampling and BLE Streaming
    // Runs at SAMPLE_INTERVAL_MS (10ms = 100Hz) only when a client is connected
    // -------------------------------------------------------------------------
    if (deviceConnected && (now - lastSampleTime >= SAMPLE_INTERVAL_MS)) {
        lastSampleTime = now;

        // -- Read quaternion (absolute 3D orientation from fusion algorithm) --
        // The quaternion represents a rotation in 3D space without gimbal lock.
        // Components: w (scalar), x y z (vector). Unit quaternion: w^2+x^2+y^2+z^2=1
        imu::Quaternion quat = bno.getQuat();

        // -- Read linear acceleration (dynamic acceleration, gravity removed) --
        // The BNO055 subtracts the gravity vector internally using the quaternion,
        // leaving only the net dynamic force from head movement or impact.
        // Units: m/s^2
        imu::Vector<3> linAccel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);

        // -- Read gyroscope (angular velocity) --
        // Rate of rotation around each axis. Units: degrees per second (deg/s)
        imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

        // -- Build and send the data packet --
        IMUPacket packet;
        packet.qw        = (float)quat.w();
        packet.qx        = (float)quat.x();
        packet.qy        = (float)quat.y();
        packet.qz        = (float)quat.z();
        packet.linAccX   = (float)linAccel.x();
        packet.linAccY   = (float)linAccel.y();
        packet.linAccZ   = (float)linAccel.z();
        packet.gyroX     = (float)gyro.x();
        packet.gyroY     = (float)gyro.y();
        packet.gyroZ     = (float)gyro.z();
        packet.timestamp = now;

        // Cast the struct to a raw byte array for BLE transmission
        // sizeof(IMUPacket) = 44 bytes
        // Python receives this and calls struct.unpack("<10fI", raw_data)
        pIMUCharacteristic->setValue((uint8_t*)&packet, sizeof(IMUPacket));
        pIMUCharacteristic->notify();

        // -- Local serial impact alert (debug only) --
        // Full impact analysis with history logging is done in the Python script
        // This is just so you can see high-G events in the Serial Monitor
        float accelMagnitude = sqrtf(
            packet.linAccX * packet.linAccX +
            packet.linAccY * packet.linAccY +
            packet.linAccZ * packet.linAccZ
        );

        if (accelMagnitude > IMPACT_ACCEL_THRESHOLD_MS2) {
            Serial.print("[ALERT] High accel: ");
            Serial.print(accelMagnitude);
            Serial.print(" m/s^2 (");
            Serial.print(accelMagnitude / 9.81f);
            Serial.print("g) at t=");
            Serial.print(now);
            Serial.println("ms");
        }

        // -- Verbose debug output (uncomment the block below if needed) --
        // Warning: printing at 100Hz will flood the Serial Monitor
        // Comment back out before running with the Python script connected
        /*
        Serial.print("t=");     Serial.print(now);
        Serial.print(" qw=");   Serial.print(packet.qw, 3);
        Serial.print(" qx=");   Serial.print(packet.qx, 3);
        Serial.print(" qy=");   Serial.print(packet.qy, 3);
        Serial.print(" qz=");   Serial.print(packet.qz, 3);
        Serial.print(" ax=");   Serial.print(packet.linAccX, 2);
        Serial.print(" ay=");   Serial.print(packet.linAccY, 2);
        Serial.print(" az=");   Serial.print(packet.linAccZ, 2);
        Serial.print(" gx=");   Serial.print(packet.gyroX, 1);
        Serial.print(" gy=");   Serial.print(packet.gyroY, 1);
        Serial.print(" gz=");   Serial.println(packet.gyroZ, 1);
        */
    }
}
