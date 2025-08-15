import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from world import WorldGenerator


class WorldApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procedural World Generator")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.generate_button = QPushButton("Generate New World")
        self.generate_button.clicked.connect(self.generate_and_display)
        self.layout.addWidget(self.generate_button)

        self.world_generator = WorldGenerator(size=256, octaves=6, persistence=0.5)
        self.generate_and_display()

    def generate_and_display(self):
        # Generate a new world with a random seed
        self.world_generator.generate_terrain(seed=np.random.randint(0, 10000))

        # Get the image data from the WorldGenerator instance
        image_data = self.world_generator.get_image_data()

        # Convert NumPy array to QImage for display in PyQt
        height, width, channel = image_data.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image_data.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 1))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WorldApp()
    window.show()
    sys.exit(app.exec_())
