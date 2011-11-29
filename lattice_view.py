import sys
from lattice_model import LatticeModel
from PySide import QtCore, QtGui
from lattice_widget import LatticeWidget

class LatticeView(QtGui.QWidget):
    def __init__(self, lattice_model, timer_delay = 0, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

        self.lattice_model = lattice_model
        self.lattice_widget = LatticeWidget(self.lattice_model)

        self.time = 0
        self.time_label = QtGui.QLabel("Time")
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_label.setMargin(2)

        layout = QtGui.QFormLayout()
        layout.addWidget(self.time_label)
        layout.addWidget(self.lattice_widget)

        timer = QtCore.QTimer(self)
        self.connect(timer, QtCore.SIGNAL("timeout()"), self.update_model)
        timer.start(timer_delay)

        self.setLayout(layout)
        self.setWindowTitle(self.tr("Lattice gas"))

    def update_model(self):
        for x in xrange(20):
            self.lattice_model.update()
        self.lattice_widget.update()
        self.time_label.setText(self.tr("Time " + str(self.time)))
        self.time += 1
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    size = 600
    model = LatticeModel(size, size*size)
    model.cells[1*size/2:size, 0:size/2] = 0
    model.cells[0, :] = 0
    model.cells[size - 1, :] = 0
    model.cells[:, 0] = 0
    model.cells[:, size - 1] = 0
#    model.cells[0:(2*size/5), :] = 0
#    model.cells[(3*size/5):size, :] = 0
#    model.cells[:, 0:(2*size/5)] = 0
#    model.cells[:, (3*size/5):size] = 0
#    model.cells[0, 1] = 0b0001
#    model.cells[0, 9] = 0b0100
    view = LatticeView(model, 10)
    view.show()
    sys.exit(app.exec_())
