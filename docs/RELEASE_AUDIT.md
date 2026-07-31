# Release Audit Checklist

- [ ] Confirm repository-level license.
- [ ] Confirm no model/checkpoint files are present.
- [ ] Confirm no generated result/log directories are present.
- [ ] Confirm no machine-local absolute paths remain in default commands.
- [ ] Run `colcon build --symlink-install`.
- [ ] Run a short smoke training/evaluation job.
