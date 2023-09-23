import numpy as np
from pathlib import Path
import os

relation_dict = {'unsignedinteger': 'int',
                 'shortfloat': 'float',
                 'int': 'int'}

def find_first_entry_containing_header(
    list_of_attributes,
    header,
    dtype= np.float32
    ):
    line = list_of_attributes[np.char.find(list_of_attributes, header)>=0][0]
    if dtype == np.float32:
        return np.float32(line.replace('\n', '').split(':=')[-1])
    elif dtype == str:
        return (line.replace('\n', '').split(':=')[-1].replace(' ', ''))
    elif dtype == int:
        return int(line.replace('\n', '').split(':=')[-1].replace(' ', ''))
    
def get_projections_spacing_radius(headerfile: str):
    with open(f'{headerfile}_tot_w1.h00') as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    num_proj = find_first_entry_containing_header(headerdata, 'total number of images', int)
    proj_dim1 = find_first_entry_containing_header(headerdata, 'matrix size [1]', int)
    proj_dim2 = find_first_entry_containing_header(headerdata, 'matrix size [2]', int)
    number_format = find_first_entry_containing_header(headerdata, 'number format', str)
    number_format= relation_dict[number_format]
    num_bytes_per_pixel = find_first_entry_containing_header(headerdata, 'number of bytes per pixel', int)
    imagefile = find_first_entry_containing_header(headerdata, 'name of data file', str)
    dtype = eval(f'np.{number_format}{num_bytes_per_pixel*8}')
    projections = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=dtype)
    projections = np.transpose(projections.reshape((num_proj,proj_dim2,proj_dim1))[:,::-1], (0,2,1))[0]
    # Get distance
    dx = find_first_entry_containing_header(headerdata, 'scaling factor (mm/pixel) [1]', np.float32) / 10 # to cm
    dz = find_first_entry_containing_header(headerdata, 'scaling factor (mm/pixel) [2]', np.float32) / 10 # to cm
    # Get radius
    with open(f'{headerfile}.res') as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    radius = float(find_first_entry_containing_header(headerdata, 'UpperEneWindowTresh:', str).split(':')[-1])
    return projections, (dx, dz), radius 
    