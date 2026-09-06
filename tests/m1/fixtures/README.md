# M1 feasibility fixtures

These small projects are ordinary CUDA source/build inputs. After their hashes
enter `docs/m1/fixtures.json`, a qualifying compatibility attempt uses them
byte-for-byte unchanged. Selecting cuda4AS through a compiler path, toolchain
file, install prefix, or documented environment is allowed; editing these files
for the candidate is not.

- `cmake-vector-add` exercises normal CMake CUDA-language detection, host and
  device compilation, native link, launch, synchronization, and exact output.
- `cmake-device-link` adds three CUDA translation units, an out-of-line device
  call across translation units, relocatable device code, and device link.

Both write a deterministic binary path supplied as their first argument and
also validate every element internally. The enrolled NVIDIA fixture run is a
source/reference sanity check only and never becomes an Apple-GPU pass.
