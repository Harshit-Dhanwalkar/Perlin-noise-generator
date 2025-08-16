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
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from world_2d import WorldGenerator


class WorldApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procedural World Generator")
        self.setGeometry(100, 100, 1024, 768)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.main_layout.addWidget(self.image_label, 1)

        control_panel_layout = QVBoxLayout()
        control_panel_layout.setContentsMargins(10, 10, 10, 10)

        form_layout = QFormLayout()

        # Seed Input
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Enter seed (or leave blank for random)")
        self.seed_input.textChanged.connect(self.generate_and_display)
        form_layout.addRow("Seed:", self.seed_input)

        # Octaves slider
        self.octaves_slider = QSlider(Qt.Horizontal)
        self.octaves_slider.setMinimum(1)
        self.octaves_slider.setMaximum(8)
        self.octaves_slider.setValue(6)
        self.octaves_label = QLabel(f"Octaves: {self.octaves_slider.value()}")
        self.octaves_slider.valueChanged.connect(self.update_octaves)
        form_layout.addRow(self.octaves_label, self.octaves_slider)

        # Persistence slider
        self.persistence_slider = QSlider(Qt.Horizontal)
        self.persistence_slider.setMinimum(0)
        self.persistence_slider.setMaximum(100)
        self.persistence_slider.setValue(50)
        self.persistence_label = QLabel(
            f"Persistence: {self.persistence_slider.value() / 100.0}"
        )
        self.persistence_slider.valueChanged.connect(self.update_persistence)
        form_layout.addRow(self.persistence_label, self.persistence_slider)

        # Water Threshold slider
        self.water_slider = QSlider(Qt.Horizontal)
        self.water_slider.setMinimum(0)
        self.water_slider.setMaximum(100)
        self.water_slider.setValue(30)
        self.water_label = QLabel(
            f"Water Threshold: {self.water_slider.value() / 100.0}"
        )
        self.water_slider.valueChanged.connect(self.generate_and_display)
        form_layout.addRow(self.water_label, self.water_slider)

        # Snow Threshold slider
        self.snow_slider = QSlider(Qt.Horizontal)
        self.snow_slider.setMinimum(0)
        self.snow_slider.setMaximum(100)
        self.snow_slider.setValue(20)
        self.snow_label = QLabel(f"Snow Amount: {self.snow_slider.value() / 100.0}")
        self.snow_slider.valueChanged.connect(self.generate_and_display)
        form_layout.addRow(self.snow_label, self.snow_slider)

        # Rock Probability slider
        self.rock_slider = QSlider(Qt.Horizontal)
        self.rock_slider.setMinimum(0)
        self.rock_slider.setMaximum(100)
        self.rock_slider.setValue(2)
        self.rock_label = QLabel(
            f"Rock Probability: {self.rock_slider.value() / 100.0}"
        )
        self.rock_slider.valueChanged.connect(self.generate_and_display)
        form_layout.addRow(self.rock_label, self.rock_slider)

        # Dirt Probability slider
        self.dirt_slider = QSlider(Qt.Horizontal)
        self.dirt_slider.setMinimum(0)
        self.dirt_slider.setMaximum(100)
        self.dirt_slider.setValue(5)
        self.dirt_label = QLabel(
            f"Dirt Probability: {self.dirt_slider.value() / 100.0}"
        )
        self.dirt_slider.valueChanged.connect(self.generate_and_display)
        form_layout.addRow(self.dirt_label, self.dirt_slider)

        control_panel_layout.addLayout(form_layout)

        # Horizontal layout for buttons
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

        control_panel_layout.addLayout(button_layout)
        control_panel_layout.addStretch(1)

        self.main_layout.addLayout(control_panel_layout)
        self.world_generator = WorldGenerator(size=256, octaves=6, persistence=0.5)
        self.generate_and_display()

    def update_octaves(self, value):
        self.octaves_label.setText(f"Octaves: {value}")
        self.world_generator.octaves = value
        self.generate_and_display()

    def update_persistence(self, value):
        persistence = value / 100.0
        self.persistence_label.setText(f"Persistence: {persistence}")
        self.world_generator.persistence = persistence
        self.generate_and_display()

    def generate_and_display(self):
        octaves = self.world_generator.octaves
        persistence = self.world_generator.persistence
        water_threshold = self.water_slider.value() / 100.0

        snow_threshold = self.snow_slider.value() / 100.0

        rock_probability = self.rock_slider.value() / 100.0
        dirt_probability = self.dirt_slider.value() / 100.0

        seed_text = self.seed_input.text()
        seed = int(seed_text) if seed_text.isdigit() else np.random.randint(0, 10000)

        self.water_label.setText(f"Water Threshold: {water_threshold}")
        self.snow_label.setText(f"Snow Amount: {snow_threshold}")
        self.rock_label.setText(f"Rock Probability: {rock_probability}")
        self.dirt_label.setText(f"Dirt Probability: {dirt_probability}")

        self.world_generator.generate_terrain(
            seed, water_threshold, snow_threshold, rock_probability, dirt_probability
        )
        image_data = self.world_generator.get_image_data()
        height, width, channel = image_data.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image_data.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            )
        )

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
