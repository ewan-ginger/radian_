#include <esp_now.h>
#include <WiFi.h>
#include <math.h>

// Message structure
typedef struct struct_message {
  uint32_t id;
} struct_message;

struct_message incomingMessage;

uint32_t lastPacketID = 0xFFFFFFFF;

// Interval-specific counters
uint32_t intervalReceived = 0;
uint32_t intervalExpected = 0;
int32_t rssiSum = 0;
uint64_t rssiSquaredSum = 0;
uint32_t rssiCount = 0;

unsigned long lastPrintTime = 0;
const unsigned long printInterval = 10000; // 10 seconds
uint32_t intervalID = 1; // Start from 1

// Callback when data is received
void onReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  if (len != sizeof(incomingMessage)) return;

  memcpy(&incomingMessage, incomingData, sizeof(incomingMessage));
  int32_t rssi = info->rx_ctrl->rssi;

  // Update expected count
  intervalExpected++;
  if (incomingMessage.id != lastPacketID + 1 && lastPacketID != 0xFFFFFFFF) {
    uint32_t missed = incomingMessage.id - lastPacketID - 1;
    intervalExpected += missed;
  }

  intervalReceived++;
  rssiSum += rssi;
  rssiSquaredSum += (int64_t)rssi * rssi;
  rssiCount++;

  lastPacketID = incomingMessage.id;
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed!");
    return;
  }

  esp_now_register_recv_cb(onReceive);

  Serial.println("Receiver Ready");
}

void loop() {
  unsigned long now = millis();
  if (now - lastPrintTime >= printInterval) {
    float pdr = intervalExpected > 0 ? (intervalReceived * 100.0 / intervalExpected) : 0;
    float avgRSSI = rssiCount > 0 ? (float)rssiSum / rssiCount : 0;

    float rssiStdDev = 0;
    if (rssiCount > 0) {
      float meanSquare = (float)rssiSquaredSum / rssiCount;
      rssiStdDev = sqrt(meanSquare - avgRSSI * avgRSSI);
    }

    Serial.printf("Interval %lu - PDR: %.2f%% | Avg RSSI: %.2f dBm | StdDev: %.2f dB | Packets: %lu/%lu\n",
                  intervalID, pdr, avgRSSI, rssiStdDev, intervalReceived, intervalExpected);

    // Reset for next interval
    intervalID++;
    intervalReceived = 0;
    intervalExpected = 0;
    rssiSum = 0;
    rssiSquaredSum = 0;
    rssiCount = 0;
    lastPrintTime = now;
  }
}
