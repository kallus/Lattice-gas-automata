//#include <omp.h>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "stdlib.h"
#include <stdlib.h>

//Constants for node types
#define FREE_SPACE 0
#define WALL -1
#define SOURCE -2
#define SINK -3
#define PROB 0.8

static const unsigned char random_table1[] = { 32, 16, 8, 4, 2, 1 };
static const int size_random1 = sizeof(random_table1)/sizeof(random_table1[0]);
static const unsigned char random_table2[] = { 48, 40, 36, 34, 33, 24, 20, 18, 17, 12, 10, 9, 6, 5, 3 };
static const int size_random2 = sizeof(random_table2)/sizeof(random_table2[0]);
static const unsigned char random_table3[] = { 56, 52, 50, 49, 44, 42, 41, 38, 37, 35, 28, 26, 25, 22,
                                               21, 19, 14, 13, 11, 7 };
static const int size_random3 = sizeof(random_table3)/sizeof(random_table3[0]);
static const unsigned char random_table4[] = { 60, 58, 57, 54, 53, 51, 46, 45, 43, 39, 30, 29, 27, 23, 15 };
static const int size_random4 = sizeof(random_table4)/sizeof(random_table4[0]);
static const unsigned char random_table5[] = { 62, 61, 59, 55, 47, 31 };
static const int size_random5 = sizeof(random_table5)/sizeof(random_table5[0]);

static const long nSetBits[] = { 
  0, 1, 1, 2, 1, 2, 2, 3, 
  1, 2, 2, 3, 2, 3, 3, 4, 
  1, 2, 2, 3, 2, 3, 3, 4, 
  2, 3, 3, 4, 3, 4, 4, 5, 
  1, 2, 2, 3, 2, 3, 3, 4, 
  2, 3, 3, 4, 3, 4, 4, 5, 
  2, 3, 3, 4, 3, 4, 4, 5, 
  3, 4, 4, 5, 4, 5, 5, 6
};

inline long mod(long n, long k) {
  if(n<0) return k+n;
  return n % k;
}

inline long move4(long n, long e, long s, long w) {
    return ((n & 1) | (e & 2) | (s & 4) | (w & 8));
}

inline long reverse4(long x) {
  x = (x << 2) + (x >> 2);
  return x & 15;
}

inline long move6(long nw, long ne, long e, long se, long sw, long w) {
  return ((ne & 1)
      | (e & 2)
      | (se & 4)
      | (sw & 8)
      | (w & 16)
      | (nw & 32));
}

inline long reverse6(long x) {
    return (((x << 3) + (x >> 3)) & 63);
}

static PyObject * init_cells(PyObject *self, PyObject *args) {
    PyObject *cells_python_object;
    PyObject *node_types_python_object;
    long particle_init = 0;

    if (!PyArg_ParseTuple(args, "OOl",
            &cells_python_object,
            &node_types_python_object,
            &particle_init)) {
        fprintf(stderr, "Failed to parse arguments.\n");
        Py_RETURN_NONE;
    }

    PyArrayObject *cells, *node_types;
    cells = (PyArrayObject *)PyArray_ContiguousFromAny(cells_python_object, PyArray_LONG, 2, 2);
    node_types = (PyArrayObject *)PyArray_ContiguousFromAny(node_types_python_object, PyArray_LONG, 2, 2);
    if (cells == NULL || node_types == NULL) {
        fprintf(stderr, "Invalid array object.\n");
        Py_RETURN_NONE; /* returns None to the Python side, of course. */
    }

    long iRow, iCol;
    long H = cells->dimensions[0];
    long W = cells->dimensions[1];

    for (iRow = 0; iRow < H; ++iRow) {
        long *cell = PyArray_GETPTR2(cells, iRow, 0);
        long *node_type = PyArray_GETPTR2(node_types, iRow, 0);
        for (iCol = 0; iCol < W; ++iCol, ++cell, ++node_type) {
            if (*node_type > 0) {
                long r = (rand() % 256);
                if (*node_type > r) {
                    *cell = particle_init;
                }
            }
        }
    }

    Py_XDECREF(cells);
    Py_XDECREF(node_types);

    Py_RETURN_NONE;    
}

static PyObject * update4(PyObject *self, PyObject *args) {
    PyObject *array_python_object;
    PyObject *array_python_object_temp;
    PyObject *node_types_python_object;
    PyObject *cell_colors_python_object;
    long temperature;

    if (!PyArg_ParseTuple(args, "OOOOl",
        &array_python_object,
        &array_python_object_temp,
	&node_types_python_object,
	&cell_colors_python_object,
	&temperature)) {
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
    cell_colors = (PyArrayObject *)PyArray_ContiguousFromAny(cell_colors_python_object, PyArray_UINT32, 2, 2);
    if (array == NULL || array_temp == NULL || node_types == NULL || cell_colors == NULL) {
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
        long *data = PyArray_GETPTR2(array, iRow, 0);
        long *data_temp = PyArray_GETPTR2(array_temp, iRow, 0);
        long *node_type = PyArray_GETPTR2(node_types, iRow, 0);
        for (iCol = 0; iCol < W; ++iCol, ++data, ++data_temp, ++node_type) {
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
                double r = ((double)rand())/((double)RAND_MAX);
                if(r < PROB) {
                    int n = rand() % 16;
                    (*data_temp) = n;
                    continue;
                }
	    }

            if ((*data) == 0) {
                (*data_temp) = 0;
                continue;
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
        double temp_r = rand()/((double)RAND_MAX);
        long *data = PyArray_GETPTR2(array, iRow, 0);
        long *node_type = PyArray_GETPTR2(node_types, iRow, 0);
        long mod_iRowM1 = mod(iRow-1, H);
        long mod_iRowP1 = mod(iRow+1, H);

        uint32_t *cell_color = PyArray_GETPTR2(cell_colors, iRow, 0);
        for (iCol = 0; iCol < W; ++iCol, ++data, ++cell_color, ++node_type) {
            if (iCol == 0 || iCol == (W-1)) {
                //Periodic boundary conditions
                long *n = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol);
                long *e = PyArray_GETPTR2(array_temp, iRow, mod(iCol+1, W));
                long *s = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol);
                long *w = PyArray_GETPTR2(array_temp, iRow, mod(iCol-1,W));
                (*data) = move4(*n,*e,*s,*w);
            }else{
                long *n = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol);
                long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                long *s = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol);
                long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                (*data) = move4(*n,*e,*s,*w);
            }

            /* Count the number of bits set. "You are not expected to understand this." */
            if (*node_type != WALL) {
#define COLOR_FRACTION4 63
                (*cell_color) = 255 - nSetBits[*data]*COLOR_FRACTION4;

                //Temperature
                if (temperature != 0 && *cell_color != 3 && *cell_color != 255) {
                    long this_temp_r = iCol * iRow * temp_r * *cell_color + iCol + iRow + temp_r;
                    if (this_temp_r % 100000 < temperature) {
                        switch (*cell_color) {
                        case 255 - 1*COLOR_FRACTION4:
                            (*data) = 255 - random_table1[this_temp_r % size_random1];
                            break;
                        case 255 - 2*COLOR_FRACTION4:
                            (*data) = 255 - random_table2[this_temp_r % size_random2];
                            break;
                        case 255 - 3*COLOR_FRACTION4:
                            (*data) = 255 - random_table3[this_temp_r % size_random3];
                            break;
                        default:
                            fprintf(stderr, "Bad magic %ui :(\n", *cell_color);
                            break;
                        }
                    }
                }
                (*cell_color) = (*cell_color << 24) | (*cell_color << 16) | (*cell_color << 8) | (*cell_color << 0);
            }
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
    long temperature;

    if (!PyArg_ParseTuple(args, "OOOOl",
            &array_python_object,
            &array_python_object_temp,
            &node_types_python_object,
            &cell_colors_python_object,
            &temperature)) {
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
    cell_colors = (PyArrayObject *)PyArray_ContiguousFromAny(cell_colors_python_object, PyArray_UINT32, 2, 2);
    if (array == NULL || array_temp == NULL || node_types == NULL || cell_colors == NULL) {
        fprintf(stderr, "Invalid array object.\n");
        Py_RETURN_NONE; /* returns None to the Python side, of course. */
    }

/*
    long sanity = 0, iTemp = 0;
    const int max[] = { 0, size_random1, size_random2, size_random3, size_random4, size_random5 };
    const unsigned char *tempPtr[] = { 0, random_table1, random_table2, random_table3, random_table4, random_table5 };
    for (sanity = 1; sanity < 6; sanity++) {
        unsigned char *ptr = tempPtr[sanity];
        for (iTemp = 0; iTemp < max[sanity]; iTemp++) {
            long magic = (long)ptr[iTemp];
            magic = magic - ((magic >> 1) & 0x55555555);
            magic = (magic & 0x33333333) + ((magic >> 2) & 0x33333333);
            magic = (((magic + (magic >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
            if (magic != sanity) {
                fprintf(stderr, "unsane magic: index %li, %i, does not have %li bits.\n",
                    iTemp, (int)ptr[iTemp], sanity);
                exit(1);
            }
        }
    }
*/

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
        long *data = PyArray_GETPTR2(array, iRow, 0);
        long *data_temp = PyArray_GETPTR2(array_temp, iRow, 0);
        long *node_type = PyArray_GETPTR2(node_types, iRow, 0);
        for (iCol = 0; iCol < W; ++iCol, ++data, ++data_temp, ++node_type) {

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
                double r = ((double)rand())/((double)RAND_MAX);
                if(r < PROB) {
                    int n = rand() % 64;
                    (*data_temp) |= n;
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
        double temp_r = rand()/((double)RAND_MAX);
        long *data = PyArray_GETPTR2(array, iRow, 0);
        long *node_type = PyArray_GETPTR2(node_types, iRow, 0);
        long mod_iRowM1 = mod(iRow-1, H);
        long mod_iRowP1 = mod(iRow+1, H);

//        unsigned char *cell_color = PyArray_GETPTR2(cell_colors, iRow, 0);
        uint32_t *cell_color = PyArray_GETPTR2(cell_colors, iRow, 0);
        for (iCol = 0; iCol < W; ++iCol, ++data, ++cell_color, ++node_type) {
            if (iCol == 0 || iCol == W) {
                //Periodic boundary conditions:
                if (iRow % 2 == 0) { // even row
                    long *nw = PyArray_GETPTR2(array_temp, mod_iRowM1, mod(iCol-1, W));
                    long *ne = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol);
                    long *e = PyArray_GETPTR2(array_temp, iRow, mod(iCol+1, W));
                    long *se = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol);
                    long *sw = PyArray_GETPTR2(array_temp, mod_iRowP1, mod(iCol-1, W));
                    long *w = PyArray_GETPTR2(array_temp, iRow, mod(iCol-1, W));
                    (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
                } else { // odd row
                    long *nw = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol);
                    long *ne = PyArray_GETPTR2(array_temp, mod_iRowM1, mod(iCol+1, W));
                    long *e = PyArray_GETPTR2(array_temp, iRow, mod(iCol+1, W));
                    long *se = PyArray_GETPTR2(array_temp, mod_iRowP1, mod(iCol+1, W));
                    long *sw = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol);
                    long *w = PyArray_GETPTR2(array_temp, iRow, mod(iCol-1, W));
                    (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
                }
            } else {
                if ((iRow % 2) == 0) {
                    long *nw = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol-1);
                    long *ne = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol);
                    long *sw = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol-1);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
                } else {
                    long *nw = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol);
                    long *ne = PyArray_GETPTR2(array_temp, mod_iRowM1, iCol+1);
                    long *e = PyArray_GETPTR2(array_temp, iRow, iCol+1);
                    long *se = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol+1);
                    long *sw = PyArray_GETPTR2(array_temp, mod_iRowP1, iCol);
                    long *w = PyArray_GETPTR2(array_temp, iRow, iCol-1);
                    (*data) = move6(*nw,*ne,*e,*se,*sw,*w);
                }
            }

            /* Count the number of bits set. "You are not expected to understand this." */
            if (*node_type != WALL) {
#define COLOR_FRACTION6 42
                (*cell_color) = 255 - nSetBits[*data]*COLOR_FRACTION6;

                //Temperature
                if (temperature != 0 && *cell_color != (255 - 6*COLOR_FRACTION6) && *cell_color != 255) {
                    long this_temp_r = iCol * iRow * temp_r * *cell_color + iCol + iRow + temp_r;
                    if (this_temp_r % 100000 < temperature) {
                        switch (*cell_color) {
                        case 255 - 1*COLOR_FRACTION6:
                            (*data) = 255 - random_table1[this_temp_r % size_random1];
                            break;
                        case 255 - 2*COLOR_FRACTION6:
                            (*data) = 255 - random_table2[this_temp_r % size_random2];
                            break;
                        case 255 - 3*COLOR_FRACTION6:
                            (*data) = 255 - random_table3[this_temp_r % size_random3];
                            break;
                        case 255 - 4*COLOR_FRACTION6:
                            (*data) = 255 - random_table4[this_temp_r % size_random4];
                            break;
                        case 255 - 5*COLOR_FRACTION6:
                            (*data) = 255 - random_table5[this_temp_r % size_random5];
                            break;
                        default:
                            fprintf(stderr, "Bad magic %ui :(\n", *cell_color);
                            break;
                        }
                    }
                }
                (*cell_color) = (*cell_color << 24) | (*cell_color << 16) | (*cell_color << 8) | (*cell_color << 0);
            }
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
    { "init_cells", init_cells, METH_VARARGS, "Initialize random particles." },
    { "update4", update4, METH_VARARGS, "Square lattice update." },
    { "update6", update6, METH_VARARGS, "Hexagonal lattice update." },
    { NULL, NULL, 0, NULL } /* list terminator. */
};

PyMODINIT_FUNC
initc_module(void) {
    (void)Py_InitModule("c_module", C_Module_Methods);
    import_array(); /* Numpy magic. always call this if using numpy. */
}
