from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.garden.graph import Graph, MeshLinePlot
import serial
import csv

class SerialPlotterApp(App):
    def build(self):
        self.serial_port = "/dev/ttyUSB0"  # Replace with your serial port
        self.baud_rate = 115200
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)

        self.layout = BoxLayout(orientation='vertical')

        # Labels for temperature and humidity
        self.temp_label = Label(text="Temperature: -- °C", font_size='20sp')
        self.humidity_label = Label(text="Humidity: -- %", font_size='20sp')
        self.layout.add_widget(self.temp_label)
        self.layout.add_widget(self.humidity_label)

        # Graph
        self.graph = Graph(xlabel='Time', ylabel='ADC Values', x_ticks_minor=5, x_ticks_major=25, y_ticks_major=1, y_grid_label=True, x_grid_label=True, padding=5, x_grid=True, y_grid=True, xmin=0, xmax=100, ymin=0, ymax=5)
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
        self.graph.add_plot(self.plot)
        self.layout.add_widget(self.graph)

        # Buttons
        self.button_layout = BoxLayout(size_hint_y=None, height='50sp')
        self.pause_button = Button(text="Pause", on_press=self.toggle_pause)
        self.reset_button = Button(text="Reset Graph", on_press=self.reset_graph)
        self.save_button = Button(text="Save Data", on_press=self.save_data_to_file)
        self.button_layout.add_widget(self.pause_button)
        self.button_layout.add_widget(self.reset_button)
        self.button_layout.add_widget(self.save_button)
        self.layout.add_widget(self.button_layout)

        # Data storage
        self.data = []
        self.time_data = []
        self.paused = False

        # Start the update loop
        Clock.schedule_interval(self.update_data, 0.1)

        return self.layout

    def update_data(self, dt):
        if not self.paused and self.serial.in_waiting:
            line = self.serial.readline().decode('utf-8').strip()
            values = line.split('|')
            if len(values) == 10:
                try:
                    temp = float(values[0])
                    humidity = float(values[1])
                    voltages = [float(v) for v in values[2:]]

                    # Update labels
                    self.temp_label.text = f"Temperature: {temp:.2f} °C"
                    self.humidity_label.text = f"Humidity: {humidity:.2f} %"

                    # Update plot data
                    self.time_data.append(len(self.time_data))
                    self.data.append(voltages[0])
                    if len(self.data) > 100:
                        self.data.pop(0)
                        self.time_data.pop(0)

                    # Update plot
                    self.plot.points = [(x, y) for x, y in zip(self.time_data, self.data)]

                except ValueError:
                    print("Invalid data received")

    def toggle_pause(self, instance):
        self.paused = not self.paused
        self.pause_button.text = "Resume" if self.paused else "Pause"

    def reset_graph(self, instance):
        self.data = []
        self.time_data = []
        self.plot.points = []

    def save_data_to_file(self, instance):
        filename = "data.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Temperature (°C)", "Humidity (%)"] + [f"Sensor {i} (V)" for i in range(8)])
            for t, temp, hum, *adc_values in zip(self.time_data, self.data, self.data, *self.data[2:]):
                writer.writerow([t, temp, hum] + adc_values)

    def on_stop(self):
        self.serial.close()

if __name__ == "__main__":
    SerialPlotterApp().run()