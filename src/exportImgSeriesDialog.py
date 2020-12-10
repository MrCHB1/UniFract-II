from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtOpenGL import *

class exportDialog(QDialog):
    def __init__(self):
        super(exportDialog, self).__init__()

        self.defaultWidth = 800
        self.defaultHeight = 600

        self.resize(400, 250)

        self.initUi()

        self.show()
    
    def initUi(self):
        self.title = "Export Image Series"
        self.width = QLineEdit()
        self.width.setValidator(QDoubleValidator())
