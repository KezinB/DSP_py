#include <ESPNowComm.h>

ESPNowComm espNow;

void setup()
{
    Serial.begin(115200);
    uint8_t masterMac[] = {0xXX, 0xXX, 0xXX, 0xXX, 0xXX, 0xXX};

    espNow.begin(1);                    // Set as master
    espNow.addNode(0, masterMac);       // Add self
    espNow.addNode(1, slaveMacAddress); // Add slave

    espNow.setDataCallback([](SensorData data)
                           {
                               Serial.print("Received from node: ");
                               Serial.println(data.nodeId);
                               Serial.print("Current1: ");
                               Serial.println(data.current1);
                               // Update switches based on received data
                           });
}

void loop()
{
    // Send commands to slaves
    espNow.sendCommand(1, true, false); // Set switch1 ON, switch2 OFF
    delay(1000);
}