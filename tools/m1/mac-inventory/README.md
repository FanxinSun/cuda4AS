# cuda4AS M1 Mac inventory v1

This lightweight inventory captures the current native Mac, selected Xcode/SDK,
Metal tools and device enumeration, common build tools, compression-library
metadata, and free disk space. It installs and updates nothing, uses no `sudo`,
and submits no GPU workload.

From the extracted directory, run:

```bash
chmod u+x run-inventory.sh
./run-inventory.sh
```

The script writes only beneath this extracted directory. It records missing
commands and nonzero exits in `results/<UTC>/status.tsv` and retains stdout and
stderr rather than treating a missing prerequisite as success. When complete it
prints the path, byte count, and SHA-256 of one archive under `returns/`. Return
that `.tgz` without editing or extracting it.

The small Swift helper is compiled into the run's private `tmp/` directory only
when `swiftc` is already available through `xcrun`. It enumerates Metal devices
and the default device. It does not compile Metal source, allocate GPU buffers,
or submit commands. The temporary compiler outputs are excluded from the return
archive.
