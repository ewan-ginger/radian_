#include "Display_ST77916.h"
#include "LVGL_Driver.h"
#include <esp_now.h>
#include <WiFi.h>

// Define the number of slave devices
#define NUM_SLAVES 5

/*
master mac address: d8:3b:da:a0:75:04
slave 1 mac address: 98:3d:ae:3b:46:0c
slave 2 mac address: 98:3d:ae:31:3d:30
slave 3 mac address: 98:3d:ae:38:dd:70
slave 4 mac address: 98:3d:ae:39:10:50
slave 5 mac address: 98:3d:ae:37:db:b4
*/

// Array of slave MAC addresses
uint8_t slaveAddresses[NUM_SLAVES][6] = {
    {0x98, 0x3d, 0xae, 0x3b, 0x46, 0x0c},
    {0x98, 0x3d, 0xae, 0x31, 0x3d, 0x30},
    {0x98, 0x3d, 0xae, 0x38, 0xdd, 0x70},
    {0x98, 0x3d, 0xae, 0x39, 0x10, 0x50},
    {0x98, 0x3d, 0xae, 0x37, 0xdb, 0xb4}
};

// Struct to keep track of each slave's data and status
typedef struct {
    bool connected;
    unsigned long lastReceived;
    int packetCount;
} SlaveInfo;

SlaveInfo slaveInfoArray[NUM_SLAVES];

typedef struct SensorData {
    float deviceID;
    float timeStamp;
    float battery;

    float orientX;
    float orientY;
    float orientZ;

    float accelX;
    float accelY;
    float accelZ;

    float gyroX;
    float gyroY;
    float gyroZ;

    float magX;
    float magY;
    float magZ;
} SensorData;

typedef struct Message {
    char command[10];
} Message;

// Array to store data from each slave
SensorData receivedDataArray[NUM_SLAVES];
Message msg;

volatile int totalPacketCount = 0;
unsigned long lastTime = 0;

// Function to find the slave index based on MAC address
int findSlaveIndex(const uint8_t *mac_addr) {
    for (int i = 0; i < NUM_SLAVES; i++) {
        bool match = true;
        for (int j = 0; j < 6; j++) {
            if (mac_addr[j] != slaveAddresses[i][j]) {
                match = false;
                break;
            }
        }
        if (match) return i;
    }
    return -1; // Not found
}

void IRAM_ATTR OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
    // Find which slave sent the data
    int slaveIndex = findSlaveIndex(info->src_addr);
    
    if (slaveIndex != -1) {
        memcpy(&receivedDataArray[slaveIndex], incomingData, sizeof(SensorData));
        
        // Update slave info
        slaveInfoArray[slaveIndex].lastReceived = millis();
        slaveInfoArray[slaveIndex].packetCount++;
        totalPacketCount++;
        
        // Print received data with device identifier
        SensorData data = receivedDataArray[slaveIndex];
        Serial.printf("DATA: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
            data.deviceID, data.timeStamp, data.battery, 
            data.orientX, data.orientY, data.orientZ, 
            data.accelX, data.accelY, data.accelZ, 
            data.gyroX, data.gyroY, data.gyroZ, 
            data.magX, data.magY, data.magZ);
    } else {
        Serial.println("Data received from unknown device");
    }
}

void sendCommandToAll(const char *command) {
    strcpy(msg.command, command);
    for (int i = 0; i < NUM_SLAVES; i++) {
        if (slaveInfoArray[i].connected) {
            esp_err_t result = esp_now_send(slaveAddresses[i], (uint8_t *)&msg, sizeof(msg));
            Serial.printf("Command sent to Slave %d: %s\n", i + 1, 
                          result == ESP_OK ? "Success" : "Failed");
        }
    }
}

void sendCommandToSlave(int slaveIndex, const char *command) {
    if (slaveIndex >= 0 && slaveIndex < NUM_SLAVES && slaveInfoArray[slaveIndex].connected) {
        strcpy(msg.command, command);
        esp_err_t result = esp_now_send(slaveAddresses[slaveIndex], (uint8_t *)&msg, sizeof(msg));
        Serial.printf("Command sent to Slave %d: %s\n", slaveIndex + 1, 
                      result == ESP_OK ? "Success" : "Failed");
    } else {
        Serial.printf("Invalid slave index or slave not connected: %d\n", slaveIndex + 1);
    }
}

// Callback when data is sent
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    int slaveIndex = findSlaveIndex(mac_addr);
    if (slaveIndex != -1) {
        Serial.printf("Delivery to Slave %d %s\n", slaveIndex + 1,
                     status == ESP_NOW_SEND_SUCCESS ? "succeeded" : "failed");
    }
}

void showConnectedScreen() {
    lv_obj_clean(lv_scr_act());
    lv_obj_t *scr = lv_scr_act();
    lv_obj_set_style_bg_color(scr, lv_color_black(), 0);

    lv_obj_t *circle = lv_obj_create(scr);
    lv_obj_set_size(circle, 300, 300);
    lv_obj_align(circle, LV_ALIGN_CENTER, 0, 0);
    lv_obj_set_style_radius(circle, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_border_color(circle, lv_palette_main(LV_PALETTE_GREEN), 0);
    lv_obj_set_style_border_width(circle, 5, 0);
    lv_obj_set_style_bg_opa(circle, LV_OPA_TRANSP, 0);

    lv_obj_t *label = lv_label_create(scr);
    lv_label_set_text(label, "Connected");
    lv_obj_set_style_text_color(label, lv_palette_main(LV_PALETTE_GREEN), 0);
    lv_obj_set_style_text_font(label, &lv_font_montserrat_28, 0);
    lv_obj_align(label, LV_ALIGN_CENTER, 0, 0);
}

void setup() {
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    
    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW Init Failed");
        return;
    }
    
    esp_now_register_recv_cb(OnDataRecv);
    esp_now_register_send_cb(OnDataSent);
    
    // Initialize slave info array
    for (int i = 0; i < NUM_SLAVES; i++) {
        slaveInfoArray[i].connected = false;
        slaveInfoArray[i].lastReceived = 0;
        slaveInfoArray[i].packetCount = 0;
        
        // Register each slave as a peer
        esp_now_peer_info_t peerInfo = {};
        memcpy(peerInfo.peer_addr, slaveAddresses[i], 6);
        peerInfo.channel = 0;
        peerInfo.encrypt = false;
        
        if (esp_now_add_peer(&peerInfo) == ESP_OK) {
            slaveInfoArray[i].connected = true;
            Serial.printf("Successfully added Slave %d as peer\n", i + 1);
        } else {
            Serial.printf("Failed to add Slave %d as peer\n", i + 1);
        }
    }

    Backlight_Init();
    LCD_Init();
    Lvgl_Init();
    showConnectedScreen();
}

void loop() {
    Lvgl_Loop();
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        
        if (input.startsWith("all:")) {
            // Command for all slaves: "all:command"
            String command = input.substring(4);
            sendCommandToAll(command.c_str());
        } 
        else if (input.indexOf(':') > 0) {
            // Command for specific slave: "n:command" where n is the slave number (1-5)
            int colonPos = input.indexOf(':');
            int slaveNum = input.substring(0, colonPos).toInt();
            
            if (slaveNum >= 1 && slaveNum <= NUM_SLAVES) {
                String command = input.substring(colonPos + 1);
                sendCommandToSlave(slaveNum - 1, command.c_str());
            } else {
                Serial.printf("Invalid slave number: %d\n", slaveNum);
            }
        }
        else if (input.length() > 0) {
            // Default: send to all
            sendCommandToAll(input.c_str());
        }
    }

    // Check for device timeouts and print statistics
    unsigned long currentTime = millis();
    if (currentTime - lastTime >= 1000) { // Every second
        Serial.printf("TOTAL PACKETS: %d\n", totalPacketCount);
        
        // Print individual device stats
        for (int i = 0; i < NUM_SLAVES; i++) {
            bool active = (currentTime - slaveInfoArray[i].lastReceived < 5000); // Consider active if packet received in last 5 seconds
            Serial.printf("SLAVE %d: %s, Packets: %d\n", 
                          i + 1, 
                          active ? "ACTIVE" : "INACTIVE", 
                          slaveInfoArray[i].packetCount);
            slaveInfoArray[i].packetCount = 0; // Reset individual packet counts
        }
        
        totalPacketCount = 0;
        lastTime = currentTime;
    }
    
    vTaskDelay(pdMS_TO_TICKS(100));
}