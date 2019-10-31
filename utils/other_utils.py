import numpy as np

def ReadTensor(filename, bits=32, order='C', dimensions=3):
    ''' Read a binary file with three inital integers (a cbin file).

    Parameters:
        * filename (string): the filename to read from
        * bits = 32 (integer): the number of bits in the file
        * order = 'C' (string): the ordering of the data. Can be 'C'
            for C style ordering, or 'F' for fortran style.
        * dimensions (int): the number of dimensions of the data (default:3)

    Returns:
        The data as a three dimensional numpy array.
    '''

    assert(bits == 32 or bits == 64)

    f = open(filename)

    temp_mesh = np.fromfile(f, count=dimensions, dtype='int32')

    datatype = np.float32 if bits == 32 else np.float64
    data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
    data = data.reshape(temp_mesh, order=order)
    return data