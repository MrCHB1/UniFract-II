from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from pathlib import Path
import main

class fractalEditor(QDialog):
    def __init__(self, glWidget, program, vs, fs, parent=None):
        super().__init__()

        self.glWidget = glWidget
        self.program = program
        self.vs = vs
        self.fs = fs
        self.setWindowTitle("Formula Editor")

        self.initUI()

    def initUI(self):
        self.glWidget = main.GLWidget()

        try:
            self.z = open("main.py", "r")
        except:    
            self.z = open("src/main.py", "r")

        self.f = self.z.readlines()

        self.zreal = QLineEdit(self)
        self.zreal.resize(self.zreal.width() + 20, self.zreal.height())
        self.zreal.move(5, 5)
        self.zimag = QLineEdit(self)
        self.zimag.resize(self.zimag.width() + 20, self.zimag.height())
        self.zimag.move(self.zreal.width() + 10, self.zreal.y())

        self.zreal.setText(self.f[122])
        self.zimag.setText(self.f[123])

        self.zreal.textEdited.connect(self.editFormula)
        self.zimag.textEdited.connect(self.editFormula)

        self.resetFormula = QPushButton(self)
        self.resetFormula.setText("Reset Formula")
        self.setToolTip("Press this if the program is showing black, then reset.")
        self.resetFormula.clicked.connect(self.reset)
        self.resetFormula.move(5, 40)
        self.resetFormula.resize(self.zreal.width() + self.zimag.width() + 5, self.resetFormula.height())

    def editFormula(self):
        self.f[122] = self.zreal.text()
        self.f[123] = self.zimag.text()

        with open("src/main.py", "w") as f1:
            f1.writelines(self.f)
        self.program = compileProgram(compileShader(main.vertexShader, GL_VERTEX_SHADER), compileShader(main.fragmentShader, GL_FRAGMENT_SHADER))
        self.glWidget.update()

    def reset(self):
        self.zreal.setText("znew.x = z.x*z.x + z.y*z.y + c.x;")
        self.zimag.setText("znew.y = 2.0 * z.x * z.y + c.y;")
