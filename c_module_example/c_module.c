#include "Python.h"
#include "numpy/arrayobject.h"

/* this function takes an integer number i, a list, a double d and a Numpy matrix of doubles.
   it copies the first i doubles from the array to the list,
   and multiplies each element in the array by d. */
static PyObject * c_module(PyObject *self, PyObject *args) {
    long integer_number;
    PyObject *list_object;
    double double_number;
    PyObject *array_python_object;
    PyArrayObject *array;

    if (!PyArg_ParseTuple(args, "lOdO",
            &integer_number,  /* l means long */
            &list_object,     /* O means (Python) object. note that it must be capitalized. */
            &double_number,   /* d means double */
            &array_python_object)) { /* this is also a python object */
        fprintf(stderr, "Failed to parse arguments.\n");
        Py_RETURN_NONE;
    }

    /* PyArray_XYZ specifies the data type.
       The first 2 is the minimum dimension of the array. 2 means two-dimensional array, ie. a matrix.
       The second 2 is the maximum dimension. Here we always require a matrix, so it's also 2. */
    array = (PyArrayObject *)PyArray_ContiguousFromAny(array_python_object, PyArray_DOUBLE, 2, 2);
    if (array == NULL) {
        fprintf(stderr, "Invalid array object.\n");
        /* make sure we remove our refererences to the python objects we have before returning. */
        Py_DECREF(list_object);
        Py_DECREF(array_python_object);
        Py_RETURN_NONE; /* returns None to the Python side, of course. */
    }

    int iRow, iCol;
    for (iRow = 0; iRow < array->dimensions[0]; ++iRow) {
        for (iCol = 0; iCol < array->dimensions[1]; ++iCol) {
            /* GETPTR2 is a macro to get a pointer to the given element, taking memory layout etc. into account. */
            double *data = PyArray_GETPTR2(array, iRow, iCol);
            printf("row %i col %i = %f\n", iRow, iCol, *data);
            /* we can only add Python objects to a Python list, so we need to create a PyDouble object from
               the raw double in the array. */
            if (integer_number > 0) {
                PyObject *new_double = PyFloat_FromDouble(data[iRow*array->dimensions[1] + iCol]);
                PyList_Append(list_object, new_double);
                /* IMPORTANT: need to keep track of references. if we create an object we obviously hold a reference
                   to it. if we put that object into the list, the list has another reference.
                   after putting it into the list, we are done with the object, so we must remove our own reference. */
                Py_DECREF(new_double);

                /* we can change this without the Python side ever noticing, since it's just a C long. */
                integer_number -= 1;
            }

            /* however, the numpy array memory is shared, so changes will be noticed on the Python side. */
            (*data) *= double_number;
        }
    }

    /* this is just like DECREF, but doesn't crash if the argument is NULL. */
    Py_XDECREF(array);

    Py_RETURN_NONE;
}

static PyMethodDef C_Module_Methods[] = {
    /* function name, function pointer, always METH_VARARGS (could be KEYWORDS too), documentation string. */
    { "c_module", c_module, METH_VARARGS, "C module example function." },
    { NULL, NULL, 0, NULL } /* list terminator. */
};

PyMODINIT_FUNC
initc_module(void) {
    (void)Py_InitModule("c_module", C_Module_Methods);
    import_array(); /* Numpy magic. always call this if using numpy. */
}
