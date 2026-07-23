import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


def read_vtk(mesh_path: str):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    mesh = reader.GetOutput()
    return mesh


def write_vtk(mesh, mesh_path: str):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(mesh_path)
    writer.SetInputData(mesh)
    writer.Write()


def load_vtk_efield(mesh_path):
    mesh = read_vtk(mesh_path)

    # Points (Nx3)
    points = vtk_to_numpy(mesh.GetPoints().GetData())

    # Scalars (E-field)
    point_data = mesh.GetPointData()

    array = point_data.GetArray(0)

    values = vtk_to_numpy(array)

    return points, values