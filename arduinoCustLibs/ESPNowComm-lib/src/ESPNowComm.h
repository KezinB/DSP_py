// ESPNowComm.h
#ifndef ESPNowComm_h
#define ESPNowComm_h

#include <esp_now.h>
#include <WiFi.h>

#define MAX_NODES 5
#define DATA_SIZE sizeof(SensorData)
#define CMD_SIZE sizeof(Command)

// Custom data structure for sensor data
struct SensorData
{
    uint8_t nodeId;
    bool switch1;
    bool switch2;
    int current1;
    int current2;
};

// Command structure for switch positions
struct Command
{
    uint8_t targetId;
    bool switch1;
    bool switch2;
};

class ESPNowComm
{
public:
    ESPNowComm();

    void begin(uint8_t role, uint8_t nodeId = 0);
    bool addNode(uint8_t nodeId, const uint8_t *macAddr);
    void sendSensorData(const SensorData &data);
    void sendCommand(uint8_t targetId, bool sw1, bool sw2);

    // Callback function types
    typedef void (*DataReceivedCallback)(SensorData data);
    typedef void (*CommandReceivedCallback)(Command cmd);

    void setDataCallback(DataReceivedCallback cb);
    void setCommandCallback(CommandReceivedCallback cb);

private:
    uint8_t _role; // 0 = slave, 1 = master
    uint8_t _nodeId;
    esp_now_peer_info_t _peerList[MAX_NODES];

    static DataReceivedCallback _dataCallback;
    static CommandReceivedCallback _cmdCallback;

    static void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status);
    static void onDataRecv(const uint8_t *mac_addr, const uint8_t *data, int data_len);
};

#endif