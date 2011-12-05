#include "Python.h"
#include "numpy/arrayobject.h"

long mod(long n, long k) {
  if(n<0) return k+n;
  return n % k;
}

long move4(long n, long e, long s, long w) {
  long ret = 0;
  if(n % 2 == 1) ret += 1;
  if((e>>1) % 2 == 1) ret += 2;
  if((s>>2) % 2 == 1) ret += 4;
  if((w>>3) % 2 == 1) ret += 8;
  return ret;
}

long reverse4(long x) {
  x = (x << 2) + (x >> 2);
  return x & 15;
}

long move6(long nw, long ne, long e, long se, long sw, long w) {
  long ret = 0;
  if(ne % 2 == 1) ret += 1;
  if((e>>1) % 2 == 1) ret += 2;
  if((se>>2) % 2 == 1) ret += 4;
  if((sw>>3) % 2 == 1) ret += 8;
  if((w>>4) % 2 == 1) ret += 16;
  if((nw>>5) % 2 == 1) ret += 32;
  return ret;
}

long reverse6(long x) {
  x = (x << 3) + (x >> 3);
  return x & 63;
}

long on_border(long H, long W, long y, long x) {
  if (x == 0 || y == 0 || x == W-1 || y == H-1) {
    return 1;
  }

  return 0;
}

long on_corner(long H, long W, long y, long x) {
  if ((x == 0 || x == W-1) && (y == 0 || y == H-1)) {
    return 1;
  }

  return 0;
}

long which_corner(long H, long W, long y, long x) {
  if (y == 0 && x == 0) return 1;
  if (y == 0 && x == W-1) return 2;
  if (y == H-1 && x == W-1) return 3;
  if (x == 0 && y == H-1) return 4;
  fprintf(stderr, "unexpected in which_corner");
  return 0;
}

long which_border(long H, long W, long y, long x) {
  if (y == 0) return 1;
  if (x == W-1) return 2;
  if (y == H-1) return 3;
  if (x == 0) return 4;
  fprintf(stderr, "unexpected in which_border");
  return 0;
}


static PyObject * update4(PyObject *self, PyObject *args) {
    PyObject *array_python_object;
    PyObject *array_python_object_temp;

    if (!PyArg_ParseTuple(args, "OO",
            &array_python_object,
            &array_python_object_temp)) {
        fprintf(stderr, "Failed to parse arguments.\n");
        Py_RETURN_NONE;
    }

    PyArrayObject *array;
    PyArrayObject *array_temp;
    array = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object, PyArray_LONG, 2, 2);
    array_temp = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object_temp, PyArray_LONG, 2, 2);
    if (array == NULL || array_temp == NULL) {
        fprintf(stderr, "Invalid array object.\n");
        /* make sure we remove our refererences to the python objects we have before returning. */
        Py_DECREF(array_python_object);
        Py_DECREF(array_python_object_temp);
        Py_RETURN_NONE; /* returns None to the Python side, of course. */
    }

    // read from array, write to array_temp
    // collision step
    long iRow, iCol;
    long H = array->dimensions[0];
    long W = array->dimensions[1];
    /* long particle_count = 0; */
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);
            long *data_temp = PyArray_GETPTR2(array_temp, iRow, iCol);

            /* long data_copy = *data; */
            /* long ic; */
            /* for(ic = 0; ic < 4; ic++) { */
            /*   if (data_copy % 2 == 1) particle_count++; */
            /*   data_copy = data_copy >> 1; */
            /* } */

            // cell collision
            /*if (!on_border(H, W, iRow, iCol)) {*/
              if (*data == 5) {
                (*data_temp) = 10;
              } else if (*data == 10) {
                (*data_temp) = 5;
              } else {
                (*data_temp) = *data;
              }
	      /*} else { // on border or corner
              (*data_temp) = reverse4(*data);
	      }*/
        }
    }
    
    //    fprintf(stdout,"particles: %d\n", particle_count);
    
    // read from array_temp, write to array
    // move step
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);
            //if (!on_border(H, W, iRow, iCol)) {
	    long *n = PyArray_GETPTR2(array_temp, mod(iRow-1, H), iCol);
	    long *e = PyArray_GETPTR2(array_temp, iRow, mod(iCol+1, W));
	    long *s = PyArray_GETPTR2(array_temp, mod(iRow+1,H), iCol);
	    long *w = PyArray_GETPTR2(array_temp, iRow, mod(iCol-1,W));
              (*data) = move4(*n,*e,*s,*w);
	      /*} else { // on border
              if (on_corner(H, W, iRow, iCol)) {
                corner = which_corner(H, W, iRow, iCol);
                if (corner == 1) { // NW
                  long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                  long *s = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                  (*data) = move4(0,*e,*s,0);
                } else if (corner == 2) { // NE
                  long *s = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                  long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                  (*data) = move4(0,0,*s,*w);
                } else if (corner == 3) { // SE
                  long *n = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                  long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                  (*data) = move4(*n,0,0,*w);
                } else { // SW
                  long *n = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                  long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                  (*data) = move4(*n,*e,0,0);
                }
              } else { // on border, not on corner
                border = which_border(H, W, iRow, iCol);
                if (border == 1) { // north
                  long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                  long *s = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                  long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                  (*data) = move4(0,*e,*s,*w);
                } else if (border == 2) { // east
                  long *n = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                  long *s = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                  long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                  (*data) = move4(*n,0,*s,*w);
                } else if (border == 3) { // south
                  long *n = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                  long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                  long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                  (*data) = move4(*n,*e,0,*w);
                } else { // west
                  long *n = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                  long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                  long *s = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                  (*data) = move4(*n,*e,*s,0);
                }
		}
		}*/
        }
    }

    /* this is just like DECREF, but doesn't crash if the argument is NULL. */
    Py_XDECREF(array);
    Py_XDECREF(array_temp);

    Py_RETURN_NONE;
}

static PyObject * update6(PyObject *self, PyObject *args) {
    PyObject *array_python_object;
    PyObject *array_python_object_temp;

    if (!PyArg_ParseTuple(args, "OO",
            &array_python_object,
            &array_python_object_temp)) {
        fprintf(stderr, "Failed to parse arguments.\n");
        Py_RETURN_NONE;
    }

    PyArrayObject *array;
    PyArrayObject *array_temp;
    array = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object, PyArray_LONG, 2, 2);
    array_temp = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object_temp, PyArray_LONG, 2, 2);
    if (array == NULL || array_temp == NULL) {
        fprintf(stderr, "Invalid array object.\n");
        /* make sure we remove our refererences to the python objects we have before returning. */
        Py_DECREF(array_python_object);
        Py_DECREF(array_python_object_temp);
        Py_RETURN_NONE; /* returns None to the Python side, of course. */
    }

    // read from array, write to array_temp
    // collision step
    long iRow, iCol, corner, border;
    long H = array->dimensions[0];
    long W = array->dimensions[1];

    long collide[64] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    // tripple collisions
    collide[21] = 42;
    collide[42] = 21;
    collide[19] = 37;
    collide[37] = 19;
    collide[50] = 41;
    collide[41] = 50;
    collide[22] = 13;
    collide[13] = 22;
    collide[26] = 44;
    collide[44] = 26;
    collide[25] = 52;
    collide[52] = 25;
    collide[38] = 11;
    collide[11] = 38;
    // quadruple collisions
    collide[45] = 27;
    collide[27] = 54;
    collide[54] = 45;
    // double collisions
    collide[ 9] = 18;
    collide[18] = 36;
    collide[36] =  9;

    /* long particle_count = 0; */
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);
            long *data_temp = PyArray_GETPTR2(array_temp, iRow, iCol);

            /* long data_copy = *data; */
            /* long ic; */
            /* for(ic = 0; ic < 4; ic++) { */
            /*   if (data_copy % 2 == 1) particle_count++; */
            /*   data_copy = data_copy >> 1; */
            /* } */

            // cell collision
            if (!on_border(H, W, iRow, iCol)) {
              if (collide[*data] == 0) {
                (*data_temp) = *data;
              } else {
                (*data_temp) = collide[*data];
              }
            } else { // on border or corner
              (*data_temp) = reverse4(*data);
            }
        }
    }
//    fprintf(stdout,"particles: %d\n", particle_count);

    // read from array_temp, write to array
    // move step
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);
            if (iRow % 2 == 0) { // even row
              if (!on_border(H, W, iRow, iCol)) {
                long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol-1);
                long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol-1);
                long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
              } else { // on border
                if (on_corner(H, W, iRow, iCol)) {
                  corner = which_corner(H, W, iRow, iCol);
                  if (corner == 1) { // NW
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    (*data) = move6(0,0,*e,*se,0,0);
                  } else if (corner == 2) { // NE
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol-1);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(0,0,0,*se,*sw,*w);
                  } else if (corner == 3) { // SE
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol-1);
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,*ne,0,0,0,*w);
                  } else { // SW
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    (*data) = move6(0,*ne,*e,0,0,0);
                  }
                } else { // on border, not on corner
                  border = which_border(H, W, iRow, iCol);
                  if (border == 1) { // north
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol-1);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(0,0,*e,*se,*sw,*w);
                  } else if (border == 2) { // east
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol-1);
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol-1);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,*ne,0,*se,*sw,*w);
                  } else if (border == 3) { // south
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol-1);
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,*ne,*e,0,0,*w);
                  } else { // west
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    (*data) = move6(0,*ne,*e,*se,0,0);
                  }
                }
              }
            } else { // odd row
              if (!on_border(H, W, iRow, iCol)) {
                long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol+1);
                long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol+1);
                long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
              } else { // on border
                if (on_corner(H, W, iRow, iCol)) {
                  corner = which_corner(H, W, iRow, iCol);
                  if (corner == 1) { // NW
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol+1);
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    (*data) = move6(0,0,*e,*se,*sw,0);
                  } else if (corner == 2) { // NE
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(0,0,0,0,*sw,*w);
                  } else if (corner == 3) { // SE
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,0,0,0,0,*w);
                  } else { // SW
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol+1);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    (*data) = move6(*nw,*ne,*e,0,0,0);
                  }
                } else { // on border, not on corner
                  border = which_border(H, W, iRow, iCol);
                  if (border == 1) { // north
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol+1);
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(0,0,*e,*se,*sw,*w);
                  } else if (border == 2) { // east
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,0,0,0,*sw,*w);
                  } else if (border == 3) { // south
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol+1);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,*ne,*e,0,0,*w);
                  } else { // west
                    long *nw = PyArray_GETPTR2(array_temp, iRow-1, iCol);
                    long *ne = PyArray_GETPTR2(array_temp, iRow-1, iCol+1);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, iRow+1, iCol+1);
                    long *sw = PyArray_GETPTR2(array_temp, iRow+1, iCol);
                    (*data) = move6(*nw,*ne,*e,*se,*sw,0);
                  }
                }
              }
            }
        }
    }

    /* this is just like DECREF, but doesn't crash if the argument is NULL. */
    Py_XDECREF(array);
    Py_XDECREF(array_temp);

    Py_RETURN_NONE;
}

static PyMethodDef C_Module_Methods[] = {
    /* function name, function polonger, always METH_VARARGS (could be KEYWORDS too), documentation string. */
    { "update4", update4, METH_VARARGS, "Square lattice update." },
    { "update6", update6, METH_VARARGS, "Hexagonal lattice update." },
    { NULL, NULL, 0, NULL } /* list terminator. */
};

PyMODINIT_FUNC
initc_module(void) {
    (void)Py_InitModule("c_module", C_Module_Methods);
    import_array(); /* Numpy magic. always call this if using numpy. */
}
