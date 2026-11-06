# v1.1.1 MATLAB functions, renamed

`track_v111.m`, `closure_v111.m`, `decomposePair_v111.m` and `decompose_v111.m`
are the files of tag `v1.1.1` with only the function name on the first line
changed (`track` -> `track_v111`, ...). `tests/test_ncvd.m` calls them next to
the current `ncvd.*` functions to check that every default call of 1.1.2
returns what 1.1.1 returns (to 1e-12 relative). The Python test
`test_matlab_v111_copies_are_the_tag` restores the original name and checks the
git blob hash against the tag, so any other edit fails the Python suite.

The copies call the current package for their helpers (`ncvd.decompose`,
`ncvd.realify`, `ncvd.physicalModes`, `ncvd.windowEdges`); those helpers are
either unchanged since 1.1.1 or have their default path checked by the
`decompose`/`decomposePair` comparisons.
