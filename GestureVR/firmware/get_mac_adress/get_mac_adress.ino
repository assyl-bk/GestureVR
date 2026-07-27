#include <WiFi.h>

void setup() {
  Serial.begin(115200);
  delay(1000); // let Serial and WiFi hardware settle before reading MAC

  WiFi.mode(WIFI_STA);
  delay(500); // WiFi.macAddress() can return all zeros if read too early

  Serial.print("This ESP32's MAC address: ");
  Serial.println(WiFi.macAddress());
}

void loop() {
  // Print again every 3 seconds in case you missed it the first time
  delay(3000);
  Serial.print("MAC address: ");
  Serial.println(WiFi.macAddress());
}