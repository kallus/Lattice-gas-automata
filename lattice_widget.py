from PySide import QtCore, QtGui

class LatticeWidget(QtGui.QWidget):
    def __init__(self, lattice_model, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.lattice_model = lattice_model

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        diameter = (self.width() / float(self.lattice_model.size))
        
        painter.begin(self)
        painter.fillRect(event.rect(), QtCore.Qt.white)
        painter.end()

        for row in xrange(self.lattice_model.size):
            for col in xrange(self.lattice_model.size):
                if self.lattice_model.cells[row, col] <> 0:
                    painter.begin(self)
                    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                    w_x = diameter/2.0 + (float(col) / self.lattice_model.size)*self.width()
                    w_y = diameter/2.0 + (float(row) / self.lattice_model.size)*self.height()
                    painter.translate(w_x, w_y)
                    pen = painter.pen()
                    pen.setColor(QtCore.Qt.black)
                    pen.setBrush(QtCore.Qt.black)
                    painter.setPen(pen)
                    painter.drawEllipse(QtCore.QRect(-diameter/2.0, -diameter/2.0, diameter, diameter))
                    painter.end()
                    
    def minimumSizeHint(self):
        return QtCore.QSize(200, 200)

    def sizeHint(self):
        return QtCore.QSize(400, 400)
