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

        points1 = []
        points2 = []
        points3 = []
        points4 = []

        mul_x = float(self.width())/float(self.lattice_model.size)
        mul_y = float(self.height())/float(self.lattice_model.size)
        for row in xrange(self.lattice_model.size):
            w_y = float(row) * mul_y
            for col in xrange(self.lattice_model.size):
                if self.lattice_model.cells[row, col] == 0:
                    continue
                w_x = float(col) * mul_x
                point = QtCore.QPointF(w_x, w_y)
                if self.lattice_model.cells[row, col] == 0b0001 or \
                   self.lattice_model.cells[row, col] == 0b0010 or \
                   self.lattice_model.cells[row, col] == 0b0100 or \
                   self.lattice_model.cells[row, col] == 0b1000:
                    #rects.append(QtCore.QRectF(w_x - diameter/2.0, w_y - diameter/2.0, diameter, diameter))
                    points1.append(point)
                elif self.lattice_model.cells[row, col] == 0b0111 or \
                     self.lattice_model.cells[row, col] == 0b1011 or \
                     self.lattice_model.cells[row, col] == 0b1101 or \
                     self.lattice_model.cells[row, col] == 0b1110:
                    points3.append(point)
                elif self.lattice_model.cells[row, col] == 0b1111:
                    points4.append(point)
                else:
                    points2.append(point)

        #painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)

        color = QtGui.QColor("black")
        painter.begin(self)
        pen = painter.pen()
        pen.setColor(color)
        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawPoints(points4)
        painter.end()

        color.setRgbF(0.125, 0.125, 0.125)
        painter.begin(self)
        pen = painter.pen()
        pen.setColor(color)
        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawPoints(points3)
        painter.end()

        color.setRgbF(0.375, 0.375, 0.375)
        painter.begin(self)
        pen = painter.pen()
        pen.setColor(color)
        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawPoints(points2)
        painter.end()

        color.setRgbF(0.5, 0.5, 0.5)
        painter.begin(self)
        pen = painter.pen()
        pen.setColor(color)
        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawPoints(points1)
        painter.end()

#        painter.drawRects(rects) # QtCore.QRect(-diameter/2.0, -diameter/2.0, diameter, diameter))
#        painter.drawPoints(points)
#        painter.end()
        
    def minimumSizeHint(self):
        return QtCore.QSize(200, 200)

    def sizeHint(self):
        return QtCore.QSize(600, 600)
