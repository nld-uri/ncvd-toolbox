# Frozen copy of the v1.1.1 Python package

`ncvd/` is a byte-exact copy of `python/ncvd/` at tag `v1.1.1`
(commit `4c9624bdd9879f151f4e0f9d966c0d33a827997c`). It is loaded under the module name `ncvd_v111` by
`tests/test_release_112.py`, which runs the default calls of `track`, `ncvd`,
`ncvd_pair`, `closure`, `robustness_basis`, `deflate` and `escape` on the
synthetic records of `tests/_records.py` under both versions and requires
agreement to 1e-12 relative.

`MANIFEST.json` holds the git blob hashes of the tagged files;
`test_vendored_copy_is_the_v111_tag` recomputes them, so an edit to this copy
fails the suite. Do not edit these files.
