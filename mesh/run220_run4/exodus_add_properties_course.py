filenameExodus = "temp_mesh.exo"
minPeriod = 10.0

# ======================================================================
import sys
import numpy
import netCDF4

#Define global variables


#Length of domain (Defined in geometry.jou)
L = 300.0e3

#min and max cell size (defined in mesh_geometry.jou)
min_cell_size = 0.5e3
max_cell_size = 20.0e3
#max_cell_size = 10.0e3


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Cell size based on analytical function of vertex coordinates.
def getCellSize(points):
    #coordinates of origin
    origin = (0.0, 0.0)

    #compute x y distance from origin
    r = ((points[:,0] - origin[0])**2 + (points[:,1] - origin[1])**2)**0.5

    #compute cell size
    cell_size = min_cell_size + (max_cell_size-min_cell_size)*(2*r/L)
    
    return cell_size



# ----------------------------------------------------------------------
# Get coordinates of points from ExodusII file.
exodus = netCDF4.Dataset(filenameExodus, 'a')
points = exodus.variables['coord'][:].transpose()
cellSizeFn = getCellSize(points)

# Add cell size info to ExodusII file
if not 'num_nod_var' in exodus.dimensions.keys():
    exodus.createDimension('num_nod_var', 2)

    name_nod_var = exodus.createVariable('name_nod_var', 'S1',
                                       ('num_nod_var', 'len_string',))
    name_nod_var[0,:] = netCDF4.stringtoarr("cell_size_fn", 33)
    
    vals_nod_var = exodus.createVariable('vals_nod_var', numpy.float64,
                                       ('time_step', 'num_nod_var', 'num_nodes',))


time_whole = exodus.variables['time_whole']
time_whole[0] = 0.0
vals_nod_var = exodus.variables['vals_nod_var']
vals_nod_var[0,0,:] = cellSizeFn.transpose()

exodus.close()


# End of file




