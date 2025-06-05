#include <esp_now.h>
#include <WiFi.h>
#include <CodeCell.h>

CodeCell myCodeCell;

// --- Your existing global variables and structs ---
float deviceID = 1.0;
// ... (other sensor data floats) ...
float timeStamp = 0.0;
float battery = 0.0;
float orientX = 0.0; float orientY = 0.0; float orientZ = 0.0;
float accelX = 0.0; float accelY = 0.0; float accelZ = 0.0;
float gyroX = 0.0; float gyroY = 0.0; float gyroZ = 0.0;
float magX = 0.0; float magY = 0.0; float magZ = 0.0;


typedef struct SensorData {
    float deviceID;
    float timeStamp;
    float battery;
    float orientX, orientY, orientZ;
    float accelX, accelY, accelZ;
    float gyroX, gyroY, gyroZ;
    float magX, magY, magZ;
} SensorData;

typedef struct Message {
    char command[10];
} Message;

SensorData sensorData;
Message receivedMsg;

// --- State Flags ---
volatile bool device_is_connected = false;
volatile bool sensing = false;

// --- Timestamp for Idle Timeout ---
unsigned long last_activity_timestamp = 0;

// MAC Address of the ESP-NOW master device
uint8_t masterAddress[6] = {0xd8, 0x3b, 0xda, 0xa0, 0x75, 0x04}; // IMPORTANT: Replace with your master's MAC Address

// --- Configuration for Deep Sleep and Listening ---
#define CHARGING_DEEP_SLEEP_DURATION_S 60
#define DEEP_SLEEP_DURATION_S 5
#define LISTEN_CONNECT_DURATION_MS 5000
#define IDLE_TIMEOUT_BEFORE_START_MS (60 * 1000UL) // 60 seconds

// ESP-NOW Callback function when data is received
void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
    if (len >= sizeof(receivedMsg.command) -1 && len < sizeof(receivedMsg.command)) {
        memcpy(receivedMsg.command, incomingData, len);
        receivedMsg.command[len] = '\0';
    } else if (len == sizeof(receivedMsg.command)) {
         memcpy(receivedMsg.command, incomingData, sizeof(receivedMsg.command));
         // Assuming sender ensures null termination if command fills buffer
    } else {
        Serial.printf("Received data of unexpected length: %d bytes. Max command length is %d.\n", len, sizeof(receivedMsg.command)-1);
        return;
    }

    Serial.printf("Received command: '%s'\n", receivedMsg.command);

    if (strcmp(receivedMsg.command, "connect") == 0) {
        device_is_connected = true;
        sensing = false;
        last_activity_timestamp = millis();
        Serial.println("Device CONNECTED by ESP-NOW command. Awaiting 'start' command (60s timeout).");
    } else if (strcmp(receivedMsg.command, "start") == 0) {
        if (device_is_connected) {
            sensing = true;
            Serial.println("Sensing STARTED by ESP-NOW command.");
        } else {
            Serial.println("'start' command received but device not connected. Please send 'connect' first.");
        }
    } else if (strcmp(receivedMsg.command, "stop") == 0) {
        sensing = false;
        device_is_connected = false;
        Serial.println("Sensing STOPPED and device DISCONNECTED by ESP-NOW command. Will check for 'connect' or sleep.");
    } else {
        Serial.printf("Unknown command received: %s\n", receivedMsg.command);
    }
}

// Function to send sensor data
void sendSensorData() {
    sensorData.deviceID = deviceID;
    sensorData.timeStamp = millis();
    sensorData.battery = myCodeCell.BatteryLevelRead();

    myCodeCell.Motion_RotationRead(orientX, orientY, orientZ);
    sensorData.orientX = orientX; sensorData.orientY = orientY; sensorData.orientZ = orientZ;

    myCodeCell.Motion_LinearAccRead(accelX, accelY, accelZ);
    sensorData.accelX = accelX; sensorData.accelY = accelY; sensorData.accelZ = accelZ;

    myCodeCell.Motion_GyroRead(gyroX, gyroY, gyroZ);
    sensorData.gyroX = gyroX; sensorData.gyroY = gyroY; sensorData.gyroZ = gyroZ;

    myCodeCell.Motion_MagnetometerRead(magX, magY, magZ);
    sensorData.magX = magX; sensorData.magY = magY; sensorData.magZ = magZ;

    esp_err_t result = esp_now_send(masterAddress, (uint8_t *)&sensorData, sizeof(sensorData));
    // Reduce serial spam during active sensing
    // if (result != ESP_OK) { Serial.print("Error sending sensor data: "); Serial.println(esp_err_to_name(result)); }
}

// Initialize ESP-NOW
void initEspNow() {
    WiFi.mode(WIFI_STA);
    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }
    Serial.println("ESP-NOW Initialized.");
    esp_now_register_recv_cb(OnDataRecv);
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, masterAddress, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add ESP-NOW peer");
    } else {
        Serial.println("ESP-NOW Peer Added.");
    }
}

// Prepare for and enter deep sleep
void enterDeepSleep() {
    // This function performs the actions needed just before sleeping
    Serial.printf("Preparing to enter deep sleep for %u seconds...\n", DEEP_SLEEP_DURATION_S);
    esp_now_deinit();
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    myCodeCell.Sleep(DEEP_SLEEP_DURATION_S); // Call the library's sleep function
    // Execution halts here until wake-up
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n----------------------");
    Serial.println("Device Waking Up / Booting...");

    myCodeCell.Init(MOTION_ROTATION + MOTION_LINEAR_ACC + MOTION_GYRO + MOTION_MAGNETOMETER);
    Serial.println("CodeCell Initialized.");

    if (myCodeCell.WakeUpCheck()) {
        Serial.println("Woke up from deep sleep (timer).");
    } else {
        Serial.println("Woke up from Power-On / Reset.");
    }

    device_is_connected = false;
    sensing = false;
    last_activity_timestamp = 0;

    initEspNow();
    Serial.println("Setup complete. Defaulting to 'Awaiting Connect' state.");
}

void loop() {
    if (!device_is_connected) {
        // --- STATE: AWAITING 'connect' COMMAND (or sleeping / USB idle) ---

        // Call Run() periodically even when not connected to keep power status updated
        // Using a low frequency like 10Hz is sufficient here.
        myCodeCell.Run(10); // <-- ADD THIS LINE HERE

        // The rest of your existing code for this block follows...

        Serial.printf("Device not connected. Listening for 'connect' command for %d ms...\n", LISTEN_CONNECT_DURATION_MS);
        unsigned long listenStartTime = millis();
        bool connect_command_received = false;

        while (millis() - listenStartTime < LISTEN_CONNECT_DURATION_MS) {
            // You could also call Run() inside this while loop if needed,
            // but calling it just before the loop and after might be enough.
            myCodeCell.Run(10); // <-- ADD THIS LINE HERE
            if (device_is_connected) {
                connect_command_received = true;
                break;
            }
            delay(20);
        }

        if (connect_command_received) {
             // Connect command was received, device_is_connected is now true.
             // Loop will iterate, and the 'else' block below will be entered.
        } else {
             // Listen window for 'connect' ended, still not connected.
             // Check power state before deciding whether to sleep or stay awake (if USB powered).

             Serial.println("Waiting briefly before reading power state...");
             delay(100); // *** ADD A DELAY HERE (e.g., 100ms) ***

             // Call Run() again just before reading the status to ensure it's as current as possible
             myCodeCell.Run(10); // <-- ADD THIS LINE HERE AGAIN (optional, but safer)

             int power_state = myCodeCell.PowerStateRead();
             Serial.printf("Power state check: %d (0=Lipo, 1=USB, 2=Init, 3=Low, 4=Charged, 5=Charging)\n", power_state);

             // ... (rest of your power state check and sleep logic)
            if (power_state == 0) {
                Serial.println("Listen window ended. No 'connect'. Running on Battery. Going to sleep.");
                enterDeepSleep();
            } else {
                Serial.println("Listen window ended. No 'connect'. USB/Charging/Low/Init state detected. Staying awake and listening for 'connect' again.");
                delay(1000);
                myCodeCell.Run(10);
            }
        }
    } else {
        // --- STATE: CONNECTED and OPERATIONAL ---
        // ... (Your existing code for the connected state, including myCodeCell.Run(50))
        if (!sensing) {;
            if (millis() - last_activity_timestamp > IDLE_TIMEOUT_BEFORE_START_MS) {
                Serial.println("Idle timeout: No 'start' command received within 60s after 'connect'. Disconnecting.");
                device_is_connected = false; // Fall back to the !device_is_connected block next loop
            }
        }

        if (device_is_connected) {
            if (myCodeCell.Run(51)) { // <-- This call is already here and correct for sensing
                if (sensing) {
                    sendSensorData();
                }
            }
        }
    }
}