/*
  Rui Santos & Sara Santos - Random Nerd Tutorials
  Complete project details at https://RandomNerdTutorials.com/get-change-esp32-esp8266-mac-address-arduino/
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.  
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

/*
master mac address: d8:3b:da:a0:75:04
slave 1 mac address: 98:3d:ae:3b:46:0c
slave 2 mac address: 98:3d:ae:31:3d:30
slave 3 mac address: 98:3d:ae:38:dd:70
slave 4 mac address: 98:3d:ae:39:10:50
slave 5 mac address: 98:3d:ae:37:db:b4
*/

#ifdef ESP32
  #include <WiFi.h>
  #include <esp_wifi.h>
#else
  #include <ESP8266WiFi.h>
#endif

void setup(){
  Serial.begin(115200);

  Serial.print("ESP Board MAC Address: ");
  #ifdef ESP32
    WiFi.mode(WIFI_STA);
    WiFi.STA.begin();
    uint8_t baseMac[6];
    esp_err_t ret = esp_wifi_get_mac(WIFI_IF_STA, baseMac);
    if (ret == ESP_OK) {
      Serial.printf("%02x:%02x:%02x:%02x:%02x:%02x\n",
                    baseMac[0], baseMac[1], baseMac[2],
                    baseMac[3], baseMac[4], baseMac[5]);
    } else {
      Serial.println("Failed to read MAC address");
    }
  #else
    Serial.println(WiFi.macAddress());
  #endif
}
 
void loop(){

}
