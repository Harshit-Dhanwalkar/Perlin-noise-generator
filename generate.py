import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
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

        form_layout = QFormLayout()

        self.octaves_slider = QSlider(Qt.Horizontal)
        self.octaves_slider.setMinimum(1)
        self.octaves_slider.setMaximum(10)
        self.octaves_slider.setValue(6)
        self.octaves_label = QLabel(f"Octaves: {self.octaves_slider.value()}")
        self.octaves_slider.valueChanged.connect(self.update_octaves)
        form_layout.addRow(self.octaves_label, self.octaves_slider)

        self.persistence_slider = QSlider(Qt.Horizontal)
        self.persistence_slider.setMinimum(0)
        self.persistence_slider.setMaximum(100)
        self.persistence_slider.setValue(50)
        self.persistence_label = QLabel(
            f"Persistence: {self.persistence_slider.value() / 100.0}"
        )
        self.persistence_slider.valueChanged.connect(self.update_persistence)
        form_layout.addRow(self.persistence_label, self.persistence_slider)

        self.layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate New World")
        self.generate_button.clicked.connect(self.generate_and_display)
        button_layout.addWidget(self.generate_button)

        self.save_button = QPushButton("Save to Image")
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.exit_button)

        self.layout.addLayout(button_layout)

        self.world_generator = WorldGenerator(size=256, octaves=6, persistence=0.5)
        self.generate_and_display()

    def update_octaves(self, value):
        self.octaves_label.setText(f"Octaves: {value}")
        self.world_generator.octaves = value

    def update_persistence(self, value):
        persistence = value / 100.0
        self.persistence_label.setText(f"Persistence: {persistence}")
        self.world_generator.persistence = persistence

    def generate_and_display(self):
        self.world_generator.generate_terrain(seed=np.random.randint(0, 10000))
        image_data = self.world_generator.get_image_data()
        height, width, channel = image_data.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image_data.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 1))

    def save_image(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "world_map.png", "PNG Image (*.png)"
        )
        if filename:
            current_pixmap = self.image_label.pixmap()
            if current_pixmap:
                current_pixmap.save(filename)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WorldApp()
    window.show()
    sys.exit(app.exec_())
