import vtk
import logging
from vtkmodules.util import numpy_support
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def sample_efield_to_volume(
        input_file: str,
        refimg: sitk.Image,
        threshold_lower: float = 200,
        scale_factor: float = 1.0,
        study=None,
        emptyElectrodeSpace: bool = False,
):
    sample_step = np.array(refimg.GetSpacing()) / scale_factor
    refimg_4x4 = np.hstack(
        [
            np.vstack([np.array(refimg.GetDirection()).reshape(3, 3), np.array([0.0, 0.0, 0.0])]),
            np.array([i for i in refimg.GetOrigin()] + [1.0]).reshape(4, 1),
        ]
    )
    # based on https://lorensen.github.io/VTKExamples/site/Python/Meshes/PointInterpolator/
    meshReader = vtk.vtkXMLUnstructuredGridReader()
    meshReader.SetFileName(input_file)

    meshReader.Update()

    if meshReader.GetOutputDataObject(0).GetPointData().GetNumberOfArrays() == 1:
        ef_array_name = meshReader.GetOutputDataObject(0).GetPointData().GetArrayName(0)
    else:
        raise ValueError(
            f"The mesh file should contain only one array, but it contains {meshReader.GetOutputDataObject(0).GetPointData().GetNumberOfArrays()} arrays."
        )

    meshReader.GetOutputDataObject(0).GetPointData().SetActiveScalars(ef_array_name)
    ef_range = meshReader.GetOutputDataObject(0).GetPointData().GetScalars().GetRange()
    logger.info(f"{ef_array_name} scalar range {ef_range}")

    # apply the inverse of the refimg direction in order to be back in image space (since vtk only works in image space)
    transform = vtk.vtkTransform()
    transform.SetMatrix(np.linalg.inv(refimg_4x4).reshape(16).tolist())
    transformField = vtk.vtkTransformFilter()
    transformField.SetTransform(transform)
    transformField.SetInputConnection(meshReader.GetOutputPort(0))
    transformField.Update()

    thresholdFilter = vtk.vtkThreshold()
    thresholdFilter.SetInputConnection(transformField.GetOutputPort(0))
    thresholdFilter.SetUpperThreshold(threshold_lower * 0.8)
    thresholdFilter.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
    thresholdFilter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, ef_array_name)
    thresholdFilter.Update()
    if thresholdFilter.GetOutputDataObject(0).GetNumberOfCells() == 0:
        logger.warning("no cells left after thresholding")
        raise ValueError("no cells left")
    else:
        logger.info(f"{thresholdFilter.GetOutputDataObject(0).GetNumberOfCells()} cells left after thresholding")

    bounds = np.array(thresholdFilter.GetOutputDataObject(0).GetBounds())
    boundsToGrid = np.zeros_like(bounds)
    refOrigin = np.array(refimg.GetOrigin())
    refSpacing = np.array(refimg.GetSpacing())

    # calculate the bound in ref spacing so that the sampling space aligns with the ref image grid
    # start point
    boundsToGrid[0::2] = np.floor((bounds[0::2] - refOrigin) // refSpacing) * refSpacing

    # shift the start point from the delunay step to the voronoi step
    # see https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html
    boundsToGrid[0::2] -= refSpacing / 2
    # end point
    boundsToGrid[1::2] = np.ceil((bounds[1::2] - refOrigin) // refSpacing) * refSpacing
    # delunay  to voronoi
    boundsToGrid[1::2] += refSpacing / 2

    boundsToGrid[0::2] += refOrigin
    boundsToGrid[1::2] += refOrigin
    logger.debug(f"bounds: {bounds} | boundsToGrid: {boundsToGrid}")

    gridSize = ((np.diff(boundsToGrid)[0::2] / refSpacing) * scale_factor).astype(int)
    gridOrigin = (boundsToGrid[0::2] + (sample_step / 2)).tolist()
    logger.info(f"grid size: {gridSize.tolist()} | grid origin: {gridOrigin} | sample step: {sample_step.tolist()}")

    if np.prod(gridSize) == 1:
        raise ValueError("null grid size")

    # create the sampling space.
    grid = vtk.vtkImageData()
    grid.SetDimensions(gridSize.tolist())
    grid.SetSpacing(sample_step)
    grid.SetOrigin(gridOrigin)
    grid.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Gaussian kernel used for the interpolation
    gaussian_kernel = vtk.vtkGaussianKernel()
    gaussian_kernel.SetKernelFootprintToNClosest()
    gaussian_kernel.SetNumberOfPoints(4)
    gaussian_kernel.SetSharpness(1)

    ug_vertex_cells = vtk.vtkUnstructuredGrid()
    ug_vertex_cells.SetPoints(thresholdFilter.GetOutputDataObject(0).GetPoints())
    ug_vertex_cells.GetPointData().SetScalars(thresholdFilter.GetOutputDataObject(0).GetPointData().GetScalars())

    # add voxel_data to unstructured grid
    for id_nr in range(thresholdFilter.GetOutputDataObject(0).GetNumberOfPoints()):
        ids = vtk.vtkIdList()
        # for id in xy:  # if want to add cells with more than one point id
        ids.InsertNextId(id_nr)
        ug_vertex_cells.InsertNextCell(1, ids)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(grid)
    interpolator.SetSourceData(ug_vertex_cells)
    interpolator.SetKernel(gaussian_kernel)
    interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.PassPointArraysOff()
    interpolator.Update()

    resampler = vtk.vtkResampleWithDataSet()
    resampler.SetSourceConnection(thresholdFilter.GetOutputPort(0))
    resampler.SetInputData(grid)
    resampler.Update()

    resampled_img = convert_vtk_to_sitk(resampler.GetOutput(), interpolator.GetOutput().GetPointData().GetScalars())

    interp_img = convert_vtk_to_sitk(interpolator.GetOutput(), interpolator.GetOutput().GetPointData().GetScalars())

    # vtkREampleWithDataSet obeys the surface of teh mesh, so the inside of the electrode
    # will be empty. This is why we still used the resampling with the gaussian kernel (old way)
    # then we paste the leaking field in the space of the electrode.

    # first find the electrode (study dependent, it depends
    # wether the electrode volume is included (LKP) or not (CLF)
    # in the simulation domain in Comsol)
    if not emptyElectrodeSpace:
        if study == "L":
            mergeElecElems = ki_et_findElectrodes(transformField, boundsToGrid)
        if study == "C":
            mergeElecElems = clf_findElectrodes(transformField, boundsToGrid, sample_step)
        if study == None:
            raise ValueError("study missing")

        # select the points on the resampling grid which are within the electrode
        pointSelector = vtk.vtkSelectEnclosedPoints()
        pointSelector.SetInputData(grid)
        pointSelector.SetSurfaceConnection(mergeElecElems.GetOutputPort(0))
        pointSelector.CheckSurfaceOff()  # todo fix maybe
        pointSelector.Update()

        # write the mask
        # pointSelect_writer = vtk.vtkMetaImageWriter()
        # pointSelect_writer.SetFileName(os.path.join(base_dir, "tmp_electrodeMask.mha"))
        # pointSelect_writer.SetInputConnection(pointSelector.GetOutputPort(0))
        # pointSelect_writer.Write()

        # some issue here
        pointSel_img = convert_vtk_to_sitk(pointSelector.GetOutput(),
                                           pointSelector.GetOutput().GetPointData().GetArray("SelectedPoints"))

        out_img = sitk.Cast(resampled_img, sitk.sitkFloat64) + (
                sitk.Cast(interp_img, sitk.sitkFloat64) * sitk.Cast(pointSel_img == 1, sitk.sitkFloat64)
        )
    else:
        out_img = sitk.Cast(resampled_img, sitk.sitkFloat64)

    out_img.SetDirection(refimg.GetDirection())
    out_img.SetOrigin(refimg_4x4[:3, 3] + np.dot(refimg_4x4[:3, :3], gridOrigin))
    return out_img


def ki_et_findElectrodes(meshReader, boundsToGrid):
    #################################################
    # BEGIN KI_ET SPECIFIC
    #################################################
    import numpy as np
    import vtk

    ###### At this point we can see we have some field leaking within the electrode in the resampled volume, because of the gaussian kernel. The idea is now to create a mask defining the electrode in order to apply it to the efield volume.

    meshSurface = vtk.vtkGeometryFilter()
    meshSurface.SetInputConnection(meshReader.GetOutputPort(0))

    # now we want to only select the part we know will be the electrode (a bit bigger than the bbox of the efield
    # to be sure that the surface of the electrode cuts the volume until the top)

    electExtract_box = vtk.vtkBox()
    electExtract_box.SetBounds(boundsToGrid.tolist())
    logger.debug(f"bbox: {boundsToGrid.tolist()}")
    electExtract = vtk.vtkExtractGeometry()
    electExtract.SetInputConnection(meshSurface.GetOutputPort(0))
    electExtract.SetImplicitFunction(electExtract_box)
    electExtract.ExtractInsideOn()
    electExtract.ExtractBoundaryCellsOn()
    electExtract.Update()

    debugWriter = vtk.vtkXMLUnstructuredGridWriter()
    # debugWriter = vtk.vtkXMLPolyDataWriter()
    debugWriter.SetFileName("debug.vtu")
    debugWriter.SetInputConnection(meshReader.GetOutputPort(0))
    debugWriter.Write()

    # we still have the surface of the Peri-electrode space, which is a domain change in comsol,
    # so we isolate that and the electrode surface by connectivity
    electConnectivity = vtk.vtkConnectivityFilter()
    electConnectivity.SetInputConnection(electExtract.GetOutputPort(0))
    electConnectivity.SetExtractionModeToAllRegions()
    electConnectivity.ColorRegionsOn()
    electConnectivity.SetRegionIdAssignmentMode(electConnectivity.CELL_COUNT_ASCENDING)
    electConnectivity.Update()

    # unfortunately selecting by cell count was not possible, so we go for OBBs.
    elemCount = electConnectivity.GetNumberOfExtractedRegions()
    logger.info(f"{elemCount} elements found")
    obbSmallAxis = list()
    obbCenters = list()
    for thisElemID in range(elemCount):
        # select one of the element based on RegionId
        filterElectrode = vtk.vtkThreshold()
        filterElectrode.SetInputConnection(electConnectivity.GetOutputPort(0))
        filterElectrode.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "RegionId")
        filterElectrode.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        filterElectrode.SetLowerThreshold(thisElemID - 0.1)
        filterElectrode.SetUpperThreshold(thisElemID + 0.1)
        filterElectrode.Update()

        # prepare the result containers
        res_corner = [0, 0, 0]  # this is the corner point,
        res_max = [0, 0, 0]  # the major axis of the obb. this vector is not a norm vector
        res_mid = [0, 0, 0]  # the middle axis of the obb. this vector is not a norm vector
        res_min = [0, 0, 0]  # the minimal axis of the obb. this vector is not a norm vector
        res_size = [0, 0, 0]  #

        # calculate the obb
        obbThisElem = vtk.vtkOBBTree()
        obbThisElem.SetMaxLevel(1)
        obbThisElem.ComputeOBB(input=filterElectrode.GetOutputDataObject(0), corner=res_corner, max=res_max,
                               mid=res_mid, min=res_min, size=res_size)
        obbSmallAxis.append(np.linalg.norm(res_min))
        obbCenters.append((np.array(res_corner) + np.array([res_max, res_mid, res_min]).sum(axis=0) / 2).tolist())

    # we cluster the objects found based on the center of their OBBs (we suppose the overlapping objects from an electrode the surface between the PES and the brain, and the electrode will have centers no further than 0.5mm from one another)
    sorted_obbs = sortListByFunction(obbCenters, testDistanceBetweenPoints, 0.5)

    # for each cluster we take the object with the smallest minimal axis
    electID_list = [np.argsort(np.array(obbSmallAxis)[thisOBBgroup])[0] for thisOBBgroup in sorted_obbs]

    logger.info(f"{len(electID_list)} objects were identified as electrodes")

    # because we may have several electrodes...
    filterElectrode = list()
    electrodeSurface = list()
    closeElect = list()
    closeElectAfterSubdiv = list()
    subdivideElectrode = list()
    mergeElecElems = vtk.vtkAppendPolyData()
    for i, electID in enumerate(electID_list):
        # select the surface we now identified to be the electrode
        filterElectrode.append(vtk.vtkThreshold())
        filterElectrode[-1].SetInputConnection(electConnectivity.GetOutputPort(0))
        filterElectrode[-1].SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "RegionId")
        filterElectrode[-1].SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        filterElectrode[-1].SetLowerThreshold(electID - 0.1)
        filterElectrode[-1].SetUpperThreshold(electID + 0.1)
        filterElectrode[-1].Update()

        # convert it to polydata
        electrodeSurface.append(vtk.vtkGeometryFilter())
        electrodeSurface[-1].SetInputConnection(filterElectrode[-1].GetOutputPort(0))

        # first closing
        closeElect.append(vtk.vtkFillHolesFilter())
        closeElect[-1].SetHoleSize(3)
        closeElect[-1].SetInputConnection(electrodeSurface[-1].GetOutputPort(0))

        # subdivide it to avoid issues when closing the proximal hole (because of electExtract)
        subdivideElectrode.append(vtk.vtkAdaptiveSubdivisionFilter())
        subdivideElectrode[-1].SetMaximumEdgeLength(0.1)
        subdivideElectrode[-1].SetInputConnection(closeElect[-1].GetOutputPort(0))

        # close again just in case
        closeElectAfterSubdiv.append(vtk.vtkFillHolesFilter())
        closeElectAfterSubdiv[-1].SetHoleSize(3)
        closeElectAfterSubdiv[-1].SetInputConnection(subdivideElectrode[-1].GetOutputPort(0))

        mergeElecElems.AddInputConnection(closeElectAfterSubdiv[-1].GetOutputPort(0))

    mergeElecElems.Update()

    return mergeElecElems
    #################################################
    ### END OF KI_ET SPECIFIC
    #################################################


def clf_findElectrodes(meshReader, boundsToGrid, sample_step):
    #################################################
    # CLF SPECIFIC
    #################################################
    import numpy as np
    import vtk

    meshConnectivity = vtk.vtkConnectivityFilter()
    meshConnectivity.SetInputConnection(meshReader.GetOutputPort(0))
    meshConnectivity.SetExtractionModeToAllRegions()
    meshConnectivity.ColorRegionsOn()
    meshConnectivity.SetRegionIdAssignmentMode(meshConnectivity.CELL_COUNT_DESCENDING)

    thresConnectivity = vtk.vtkThreshold()
    thresConnectivity.SetInputConnection(meshConnectivity.GetOutputPort(0))
    thresConnectivity.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "RegionId")
    thresConnectivity.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_LOWER)
    thresConnectivity.SetLowerThreshold(0.1)

    thresSurface = vtk.vtkGeometryFilter()
    thresSurface.SetInputConnection(thresConnectivity.GetOutputPort(0))

    boxBounds = boundsToGrid + (np.array([-1, 1, -1, 1, -1, 1]) * 10 * np.repeat(sample_step, 2))
    electExtract_box = vtk.vtkBox()
    electExtract_box.SetBounds(boxBounds.tolist())

    electExtract = vtk.vtkExtractGeometry()
    electExtract.SetInputConnection(thresSurface.GetOutputPort(0))
    electExtract.SetImplicitFunction(electExtract_box)
    electExtract.ExtractInsideOn()
    electExtract.ExtractBoundaryCellsOn()

    elecExtr_surface = vtk.vtkGeometryFilter()
    elecExtr_surface.SetInputConnection(electExtract.GetOutputPort(0))

    elecNormals = vtk.vtkPolyDataNormals()
    elecNormals.SplittingOff()
    elecNormals.SetInputConnection(elecExtr_surface.GetOutputPort(0))

    thresNormals = vtk.vtkThreshold()
    thresNormals.SetInputConnection(elecNormals.GetOutputPort(0))
    thresNormals.SetInputArrayToProcess(
        0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "vtkDataSetAttributes::NORMALS"
    )
    thresNormals.SetSelectedComponent(2)  # threshold only the component in Z

    thresNormals.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    thresNormals.SetLowerThreshold(-0.9)
    thresNormals.SetUpperThreshold(0.9)  # normals are unit vectors, we thus remove those which are 90% +-Z

    elecSurface = vtk.vtkGeometryFilter()
    elecSurface.SetInputConnection(thresNormals.GetOutputPort(0))
    # first closing
    closeElect = vtk.vtkFillHolesFilter()
    closeElect.SetHoleSize(3)
    closeElect.SetInputConnection(elecSurface.GetOutputPort(0))

    # subdivide it to avoid issues when closing the proximal hole (because of electExtract)
    subdivideElectrode = vtk.vtkAdaptiveSubdivisionFilter()
    subdivideElectrode.SetMaximumEdgeLength(0.1)
    subdivideElectrode.SetInputConnection(closeElect.GetOutputPort(0))

    # close again just in case
    closeElectAfterSubdiv = vtk.vtkFillHolesFilter()
    closeElectAfterSubdiv.SetHoleSize(3)
    closeElectAfterSubdiv.SetInputConnection(subdivideElectrode.GetOutputPort(0))
    closeElectAfterSubdiv.Update()

    return closeElectAfterSubdiv
    #################################################


def sortListByFunction(in_list, func, *args):
    import numpy as np

    tmp = in_list
    tmp_np = np.array(tmp)
    res = list()
    duplMask = np.array([[func(i, j, args) for i in tmp] for j in tmp]).squeeze()
    for i, thisNode in enumerate(tmp):
        thisNodeDupl = np.sum(duplMask[duplMask[:, i], :], axis=0) > 0
        # res.append(np.where(tmp_np[thisNodeDupl])[0].tolist())
        res.append(np.where(thisNodeDupl)[0].tolist())
        duplMask[np.where(thisNodeDupl)] = False
    return list(filter(None, res))


def testDistanceBetweenPoints(pointA, pointB, max_distance):
    import numpy as np

    return np.linalg.norm(np.array(pointB) - np.array(pointA)) <= max_distance


def convert_vtk_to_sitk(vtk_img, vtk_array):
    np_array = numpy_support.vtk_to_numpy(vtk_array)

    # Get dimensions (VTK uses x, y, z)
    dims = vtk_img.GetDimensions()

    # Reshape properly (VTK is flat, z fastest or slowest depends → correct is z, y, x)
    np_array = np_array.reshape(dims[2], dims[1], dims[0])

    sitk_image = sitk.GetImageFromArray(np_array)

    sitk_image.SetSpacing(vtk_img.GetSpacing())
    sitk_image.SetOrigin(vtk_img.GetOrigin())

    direction = vtk_img.GetDirectionMatrix()
    direction_np = [direction.GetElement(i, j) for i in range(3) for j in range(3)]
    sitk_image.SetDirection(direction_np)

    return sitk_image
