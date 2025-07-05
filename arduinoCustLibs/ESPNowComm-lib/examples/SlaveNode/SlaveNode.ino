#include <ESPNowComm.h>
#include <ACS712.h>

ESPNowComm espNow;
ACS712 currentSensor1(34, 5.0, 4095, 185); // Example parameters
ACS712 currentSensor2(35, 5.0, 4095, 185);

void setup()
{
    Serial.begin(115200);
    uint8_t slaveMac[] = {0xXX, 0xXX, 0xXX, 0xXX, 0xXX, 0xXX};

    espNow.begin(0, 1);                  // Set as slave with ID 1
    espNow.addNode(0, masterMacAddress); // Add master

    espNow.setCommandCallback([](Command cmd)
                              {
        digitalWrite(switch1Pin, cmd.switch1);
        digitalWrite(switch2Pin, cmd.switch2); });
}

void loop()
{
    SensorData data;
    data.nodeId = 1;
    data.switch1 = digitalRead(switch1Pin);
    data.switch2 = digitalRead(switch2Pin);
    data.current1 = currentSensor1.readCurrent();
    data.current2 = currentSensor2.readCurrent();

    espNow.sendSensorData(data);
    delay(1000);
}