from toga import App, MainWindow, Label, Box, Button

class SerialPlotter(App):
    def startup(self):
        # Create the main window
        self.main_window = MainWindow(title="Data Plotter")

        # Create labels for temperature and humidity
        self.temp_label = Label("Temperature: -- °C")
        self.humidity_label = Label("Humidity: -- %")

        # Create a box to hold the labels
        box = Box(children=[self.temp_label, self.humidity_label])

        # Add the box to the main window
        self.main_window.content = box

        # Show the main window
        self.main_window.show()

def main():
    return SerialPlotter()