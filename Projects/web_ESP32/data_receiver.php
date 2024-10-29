<?php
session_start();

// Check if the POST request has the data
if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['data'])) {
    // Retrieve the data
    $sensorData = $_POST['data'];

    // Store the received data in a session variable
    $_SESSION['latest_data'] = $sensorData;
    echo "Data received: " . $sensorData;
} elseif (isset($_SESSION['latest_data'])) {
    // Return the latest data if accessed via GET
    echo $_SESSION['latest_data'];
} else {
    echo "No data received.";
}
?>
