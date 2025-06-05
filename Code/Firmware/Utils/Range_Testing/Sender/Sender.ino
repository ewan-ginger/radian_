#include <esp_now.h>
#include <WiFi.h>
#include <CodeCell.h>

CodeCell myCodeCell;

// Receiver MAC Address - update this to match your receiver
uint8_t receiverMAC[] = {0xd8, 0x3b, 0xda, 0xa0, 0x75, 0x04}; // Replace with receiver MAC

typedef struct struct_message {
  uint32_t id;
} struct_message;

struct_message outgoingMessage;
uint32_t packetID = 0;

void setup() {
  Serial.begin(115200);

  myCodeCell.Init(LIGHT);
  Serial.println("CodeCell Initialized.");

  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("Transmitter Ready");
}

void loop() {
  outgoingMessage.id = packetID;
  esp_err_t result = esp_now_send(receiverMAC, (uint8_t *)&outgoingMessage, sizeof(outgoingMessage));

  if (result == ESP_OK) {
    Serial.printf("Sent packet #%lu\n", packetID);
  } else {
    Serial.println("Send error");
  }

  packetID++;
  delay(20);  // Send one packet per second
}
