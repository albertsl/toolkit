import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QMessageBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the layout
        layout = QVBoxLayout()

        # add a text label to the layout
        label = QLabel("Hello World!")
        layout.addWidget(label)

        # add a selector to the layout
        combo = QComboBox()
        combo.addItems(["Hello", "World", "!"])
        layout.addWidget(combo)

        self.combo = combo

        # add a button to the layout
        button = QPushButton('Yes')
        button.clicked.connect(self.on_button_click)
        layout.addWidget(button)

        # create a widget to hold the layout
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def on_button_click(self):
        # Create a message box
        message_box = QMessageBox()

        # Set the text and title of the message box
        message_box.setText("Welcome to Hello World!")
        message_box.setWindowTitle("Welcome" + self.combo.currentText())

        # Add buttons to the message box
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # Execute the message box and get the result
        result = message_box.exec_()

        # Check the result of the message box
        if result == QMessageBox.Ok:
            print("Ok button clicked!")
        else:
            print("Cancel button clicked!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())