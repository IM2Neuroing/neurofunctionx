# neurofunctionx

Bottom layer of the neuro / DBS stack: reusable helper functions and portable
DBS-simulation logic that the higher packages (`neurodatax`, `neurobuildx`,
`neuromodex`, `neuromodex_thal`) build on. It has no BIDS, config or model
dependencies of its own.

## Layout

- `core/` — `BaseProcessor` (shared logging / timing base class) and
  `ImageSpace` (the image-space enum).
- `subject/` — `SubjectDataBase`, the abstract base for all subject-data
  containers (subject name, data dir, versioned `files` access).
- `io/` — file, image and mesh I/O: `data_helper`, `socket_helper`,
  `sitk/` (SimpleITK load/save and image transforms), `vtk/` (mesh I/O,
  transforms, e-field-to-volume sampling).
- `comsol/` — COMSOL coordinate/conductivity handling, electrode setup,
  trajectory desorientation.
- `simulation/` — `ElectrodeTrajectoryExtractor` and
  `extract_electrode_trajectories` (electrode fitting on direct images),
  `ConductivityMapper`, and `ComsolSimulationRunner` (the portable
  Apptainer/MATLAB e-field run core). BIDS-facing orchestration such as
  `ElectrodeHandler` lives in `neurodatax`, not here.

## Dependency position

```
neurofunctionx  ->  neurodatax  ->  { neurobuildx, neuromodex ->  neuromodex_thal }
```

Everything depends on `neurofunctionx`; `neurofunctionx` depends on nothing in
the stack.
