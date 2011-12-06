//#include <omp.h>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "stdlib.h"

//Constants for node types
#define FREE_SPACE 0
#define WALL -1
#define SOURCE -2
#define SINK -3
#define PROB 0.1

inline long mod(long n, long k) {
  if(n<0) return k+n;
  return n % k;
}

inline long move4(long n, long e, long s, long w) {
  long ret = 0;
  if(n % 2 == 1) ret += 1;
  if((e>>1) % 2 == 1) ret += 2;
  if((s>>2) % 2 == 1) ret += 4;
  if((w>>3) % 2 == 1) ret += 8;
  return ret;
}

inline long reverse4(long x) {
  x = (x << 2) + (x >> 2);
  return x & 15;
}

inline long move6(long nw, long ne, long e, long se, long sw, long w) {
  long ret = 0;
  if(ne % 2 == 1) ret += 1;
  if((e>>1) % 2 == 1) ret += 2;
  if((se>>2) % 2 == 1) ret += 4;
  if((sw>>3) % 2 == 1) ret += 8;
  if((w>>4) % 2 == 1) ret += 16;
  if((nw>>5) % 2 == 1) ret += 32;
  return ret;
}

inline long reverse6(long x) {
  x = (x << 3) + (x >> 3);
  return x & 63;
}

inline long on_border(long H, long W, long y, long x) {
  if (x == 0 || y == 0 || x == W-1 || y == H-1) {
    return 1;
  }

  return 0;
}

inline long on_corner(long H, long W, long y, long x) {
  if ((x == 0 || x == W-1) && (y == 0 || y == H-1)) {
    return 1;
  }

  return 0;
}

inline long which_corner(long H, long W, long y, long x) {
  if (y == 0 && x == 0) return 1;
  if (y == 0 && x == W-1) return 2;
  if (y == H-1 && x == W-1) return 3;
  if (x == 0 && y == H-1) return 4;
  fprintf(stderr, "unexpected in which_corner");
  return 0;
}

inline long which_border(long H, long W, long y, long x) {
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
    PyObject *node_types_python_object;

    if (!PyArg_ParseTuple(args, "OOO",
        &array_python_object,
        &array_python_object_temp,
        &node_types_python_object)) {
          fprintf(stderr, "Failed to parse arguments.\n");
          Py_RETURN_NONE;
    }

    PyArrayObject *array;
    PyArrayObject *array_temp;
    PyArrayObject *node_types;

    array = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object, PyArray_LONG, 2, 2);
    array_temp = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object_temp, PyArray_LONG, 2, 2);
    node_types = (PyArrayObject *)PyArray_ContiguousFromAny(node_types_python_object, PyArray_LONG, 2, 2);
    if (array == NULL || array_temp == NULL || node_types == NULL) {
        fprintf(stderr, "Invalid array object.\n");
        Py_RETURN_NONE; /* returns None to the Python side, of course. */
    }

    // COLLISIONS AND OTHER SINGLE-NODE OPERATIONS (sources, etc)
    // read from array, write to array_temp
    long iRow, iCol;
    long H = array->dimensions[0];
    long W = array->dimensions[1];
    
    #pragma omp parallel for shared(array, array_temp, H, W, iRow) private (iCol)
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);
            long *data_temp = PyArray_GETPTR2(array_temp, iRow, iCol);
            long *node_type = PyArray_GETPTR2(node_types, iRow, iCol);

            // cell collision
            if (*data == 5) {
              (*data_temp) = 10;
            } else if (*data == 10) {
              (*data_temp) = 5;
            } else {
              (*data_temp) = *data;
            }

            //Obstacle collision
            if (WALL == *node_type) {
              (*data_temp) = reverse4(*data_temp);
            }
	    
	    //Source
	    if (SOURCE == *node_type) {
	      double rand = drand48();
	      if(rand < PROB){
		(*data_temp) = 15;
	      }
	    }

	    //Sink
	    if (SINK == *node_type) {
	      (*data_temp) = 0;
	    }
        }
    }
            
    // MOVEMENT
    // read from array_temp, write to array
    #pragma omp parallel for shared(array, array_temp, H, W, iRow) private (iCol)
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);

            //Periodic boundary conditions
            long *n = PyArray_GETPTR2(array_temp, mod(iRow-1, H), iCol);
            long *e = PyArray_GETPTR2(array_temp, iRow, mod(iCol+1, W));
            long *s = PyArray_GETPTR2(array_temp, mod(iRow+1,H), iCol);
            long *w = PyArray_GETPTR2(array_temp, iRow, mod(iCol-1,W));
            (*data) = move4(*n,*e,*s,*w);
        }
    }

    /* this is just like DECREF, but doesn't crash if the argument is NULL. */
    Py_XDECREF(array);
    Py_XDECREF(array_temp);
    Py_XDECREF(node_types);

    // I have no good explanation for why this is bad, but it is.
    // don't do it. doesn't seem to be leaking memory, either. /wjoel
//    Py_DECREF(array_python_object);
//    Py_DECREF(array_python_object_temp);
//    Py_DECREF(node_types_python_object);

    Py_RETURN_NONE;
}

static PyObject * update6(PyObject *self, PyObject *args) {
    PyObject *array_python_object;
    PyObject *array_python_object_temp;
    PyObject *node_types_python_object;
    PyObject *cell_colors_python_object;

    if (!PyArg_ParseTuple(args, "OOOO",
            &array_python_object,
            &array_python_object_temp,
            &node_types_python_object,
            &cell_colors_python_object)) {
        fprintf(stderr, "Failed to parse arguments.\n");
        Py_RETURN_NONE;
    }

    PyArrayObject *array;
    PyArrayObject *array_temp;
    PyArrayObject *node_types;
    PyArrayObject *cell_colors;
    array = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object, PyArray_LONG, 2, 2);
    array_temp = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object_temp, PyArray_LONG, 2, 2);
    node_types = (PyArrayObject *)PyArray_ContiguousFromAny(node_types_python_object, PyArray_LONG, 2, 2);
    cell_colors = (PyArrayObject *)PyArray_ContiguousFromAny(cell_colors_python_object, PyArray_UINT8, 2, 2);
    if (array == NULL || array_temp == NULL || node_types == NULL || cell_colors == NULL) {
        fprintf(stderr, "Invalid array object.\n");
        Py_RETURN_NONE; /* returns None to the Python side, of course. */
    }


    long iRow, iCol;
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

    // COLLISIONS AND OTHER SINGLE-NODE OPERATIONS (sources, etc)
    // read from array, write to array_temp
    #pragma omp parallel for shared(array, array_temp, H, W, iRow) private (iCol)
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);
            long *data_temp = PyArray_GETPTR2(array_temp, iRow, iCol);
            long *node_type = PyArray_GETPTR2(node_types, iRow, iCol);

            // cell collision            
            if (collide[*data] == 0) {
              (*data_temp) = *data;
            } else {
              (*data_temp) = collide[*data];
            }

            //Obstacle collision
            if (WALL == *node_type) {
              (*data_temp) = reverse6(*data_temp);
            }
	    
	    //Source
	    if (SOURCE == *node_type) {
	      double rand = drand48();
	      if(rand < PROB) {
		(*data_temp) = 63;
	      }
	    }

	    //Sink
	    if (SINK == *node_type) {
	      (*data_temp) = 0;
	    }
        }
    }
    
    // MOVEMENT
    // read from array_temp, write to array
    #pragma omp parallel for shared(array, array_temp, cell_colors, H, W, iRow) private (iCol)
    for (iRow = 0; iRow < H; ++iRow) {
        for (iCol = 0; iCol < W; ++iCol) {
            long *data = PyArray_GETPTR2(array, iRow, iCol);
            unsigned char *cell_color = PyArray_GETPTR2(cell_colors, iRow, iCol);

            //Periodic boundary conditions:

            if (iRow % 2 == 0) { // even row
              long *nw = PyArray_GETPTR2(array_temp, mod(iRow-1, H), mod(iCol-1, W));
              long *ne = PyArray_GETPTR2(array_temp, mod(iRow-1, H), iCol);
              long *e = PyArray_GETPTR2(array_temp, iRow, mod(iCol+1, W));
              long *se = PyArray_GETPTR2(array_temp, mod(iRow+1, H), iCol);
              long *sw = PyArray_GETPTR2(array_temp, mod(iRow+1, H), mod(iCol-1, W));
              long *w = PyArray_GETPTR2(array_temp, iRow, mod(iCol-1, W));
              (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
            } else { // odd row
              
              long *nw = PyArray_GETPTR2(array_temp, mod(iRow-1, H), iCol);
              long *ne = PyArray_GETPTR2(array_temp, mod(iRow-1, H), mod(iCol+1, W));
              long *e = PyArray_GETPTR2(array_temp, iRow, mod(iCol+1, W));
              long *se = PyArray_GETPTR2(array_temp, mod(iRow+1, H), mod(iCol+1, W));
              long *sw = PyArray_GETPTR2(array_temp, mod(iRow+1, H), iCol);
              long *w = PyArray_GETPTR2(array_temp, iRow, mod(iCol-1, W));
              (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
            }

            /* "You are not expected to understand this." */
            long magic = *data;
            magic = magic - ((magic >> 1) & 0x55555555);
            magic = (magic & 0x33333333) + ((magic >> 2) & 0x33333333);
            magic = (((magic + (magic >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
            (*cell_color) = magic*42;
        }
    }

    /* this is just like DECREF, but doesn't crash if the argument is NULL. */
    Py_XDECREF(array);
    Py_XDECREF(array_temp);
    Py_XDECREF(node_types);
    Py_XDECREF(cell_colors);

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
