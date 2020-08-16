from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from pathlib import Path
import main

class fractalEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("Formula Editor")

        self.initUI()

    def initUI(self):
        self.glWidget = main.GLWidget()

        self.z = open("main.py, "r")

        self.f = self.z.readlines()

        self.zreal = QLineEdit(self)
        self.zreal.resize(self.zreal.width() + 20, self.zreal.height())
        self.zreal.move(5, 5)
        self.zimag = QLineEdit(self)
        self.zimag.resize(self.zimag.width() + 20, self.zimag.height())
        self.zimag.move(self.zreal.width() + 10, self.zreal.y())

        self.zreal.setText(self.f[283])
        self.zimag.setText(self.f[284])

        self.zreal.textEdited.connect(self.editFormula)
        self.zimag.textEdited.connect(self.editFormula)

        self.resetFormula = QPushButton(self)
        self.resetFormula.setText("Reset Formula")
        self.setToolTip("Press this if the program is showing black, then reset.")
        self.resetFormula.clicked.connect(self.reset)
        self.resetFormula.move(5, 40)
        self.resetFormula.resize(self.zreal.width() + self.zimag.width() + 5, self.resetFormula.height())

    def editFormula(self):

        self.f[257] = self.zreal.text()
        self.f[258] = self.zimag.text()

        with open("src/main.py", "w") as f1:
            f1.writelines(self.f)

        if self.zreal.text == "       ":
            self.zreal.setText("        ")
        if self.zimag.text == "       ":
            self.zimag.setText("        ")

        self.glWidget = main.GLWidget()

    def reset(self):
        self.zreal.setText("        znew.x = z.x*z.x + z.y*z.y + c.x;")
        self.zimag.setText("        znew.y = 2.0 * z.x * z.y + c.y;")
