#!/usr/bin/env python3
"""Build and run the phase-0 CUDA kernel corpus; write the reference outputs.

    python3 cuda4AS/oracle/run_oracle.py [--arch sm_120] [--only vector_add,...]

For each kernel: build it with nvcc (read-only use of an existing CUDA install
-- nothing is installed anywhere), run it on the GPU in this box, write
`ref/<kernel>.bin`, and write `ref/<kernel>.json` holding the shape, dtype,
SHA-256 of the bytes, how a translated implementation must be compared against
them, and the machine and toolchain that produced them.  Build and run logs go
to `log/`.

THE MACHINE RULE.  These references are the CORRECTNESS ORACLE for phase 0b's
translator and nothing else.  No timing is recorded here and no performance
target is set from this card: an RTX 5080 is not a target envelope, it is the
only GPU the development box has.  The `machine` block is recorded so that a
reference produced on a different card is never silently mixed with these.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
REF = os.path.join(HERE, "ref")
LOG = os.path.join(HERE, "log")
BUILD = os.path.join(HERE, "build")

# Order matters: cheapest and most fundamental first, so a broken toolchain
# shows up on vector_add rather than eight kernels later.
CORPUS = [
    ("vector_add", []),
    ("saxpy", []),
    ("reduce_sum", []),
    ("matmul_tiled", []),
    ("histogram_atomics", []),
    ("shared_48kb", []),
    ("mma_inline_ptx", []),
    # Cooperative groups needs relocatable device code and the cudadevrt link.
    ("grid_sync", ["-rdc=true", "-lcudadevrt"]),
    ("double_dot", []),
]


def default_nvcc():
    """Prefer the newest toolkit installed, not whatever is first on PATH.

    /usr/bin/nvcc on this box is CUDA 12.4 and does not know sm_120, so a bare
    `which nvcc` silently picks a compiler that cannot target the card in the
    machine.  Nothing is installed or modified here -- these are read-only uses
    of toolkits that already exist (YFCE CLAUDE.md rule 2).
    """
    for cand in sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True):
        if os.access(cand, os.X_OK):
            return cand
    for env in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(env)
        if base and os.access(os.path.join(base, "bin", "nvcc"), os.X_OK):
            return os.path.join(base, "bin", "nvcc")
    return shutil.which("nvcc") or "nvcc"


def machine_block(nvcc):
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return None

    gpu = run(["nvidia-smi",
               "--query-gpu=name,driver_version,memory.total,compute_cap",
               "--format=csv,noheader"])
    ver = run([nvcc, "--version"])
    return {
        "role": "correctness oracle only -- not a performance target, not an "
                "envelope; see YFCE CLAUDE.md rule 1",
        "collected_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "kernel": " ".join(os.uname().release.split()[:1]),
        "platform": f"{os.uname().sysname} {os.uname().release}",
        "gpu": gpu,
        "nvcc": ver.splitlines()[-1] if ver else None,
        "nvcc_path": nvcc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_120",
                    help="the -arch nvcc builds for; sm_120 is the RTX 5080 in "
                         "this box. The Apple target compiles as __CUDA_ARCH__ "
                         "860 regardless (blueprint 2.1), so this flag names "
                         "the oracle card, not the design.")
    ap.add_argument("--nvcc", default=default_nvcc())
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    for d in (REF, LOG, BUILD):
        os.makedirs(d, exist_ok=True)
    wanted = set(x for x in args.only.split(",") if x)
    machine = machine_block(args.nvcc)
    print(f"oracle card: {machine['gpu']}")
    print(f"nvcc:        {machine['nvcc']}")
    print(f"arch:        {args.arch}\n")

    summary = []
    failures = 0
    for name, extra in CORPUS:
        if wanted and name not in wanted:
            continue
        cu = os.path.join(SRC, f"{name}.cu")
        exe = os.path.join(BUILD, name)
        binpath = os.path.join(REF, f"{name}.bin")
        cmd = [args.nvcc, "-O2", f"-arch={args.arch}", "-std=c++17",
               "-I", SRC, cu, "-o", exe] + extra
        with open(os.path.join(LOG, f"{name}.build.log"), "w") as lf:
            lf.write(" ".join(cmd) + "\n\n")
            lf.flush()
            b = subprocess.run(cmd, capture_output=True, text=True)
            lf.write(b.stdout + b.stderr)
        if b.returncode != 0:
            print(f"  {name:20} BUILD FAILED (see log/{name}.build.log)")
            tail = (b.stderr or b.stdout).strip().splitlines()[-4:]
            for line in tail:
                print(f"      {line}")
            summary.append({"kernel": name, "status": "build_failed",
                            "build_command": " ".join(cmd)})
            failures += 1
            continue

        with open(os.path.join(LOG, f"{name}.run.log"), "w") as lf:
            r = subprocess.run([exe, binpath], capture_output=True, text=True,
                               timeout=600)
            lf.write(r.stdout + "\n---- stderr ----\n" + r.stderr)
        if r.returncode != 0:
            print(f"  {name:20} RUN FAILED rc={r.returncode} "
                  f"(see log/{name}.run.log)")
            summary.append({"kernel": name, "status": "run_failed",
                            "returncode": r.returncode,
                            "stderr": r.stderr.strip()[:400]})
            failures += 1
            continue

        meta = None
        for line in r.stdout.splitlines():
            if line.startswith("ORACLE_JSON "):
                meta = json.loads(line[len("ORACLE_JSON "):])
        if meta is None:
            print(f"  {name:20} NO METADATA LINE")
            summary.append({"kernel": name, "status": "no_metadata"})
            failures += 1
            continue

        raw = open(binpath, "rb").read()
        digest = hashlib.sha256(raw).hexdigest()
        rec = dict(meta)
        rec["sha256"] = digest
        rec["bytes_on_disk"] = len(raw)
        rec["build_command"] = " ".join(cmd)
        rec["arch"] = args.arch
        rec["machine"] = machine
        rec["stderr"] = r.stderr.strip() or None
        with open(os.path.join(REF, f"{name}.json"), "w") as jf:
            json.dump(rec, jf, indent=2)
            jf.write("\n")
        print(f"  {name:20} ok  {len(raw):>10,} B  {digest[:16]}…  "
              f"compare={meta['compare']}")
        summary.append({"kernel": name, "status": "ok", "sha256": digest,
                        "bytes": len(raw), "compare": meta["compare"]})

    with open(os.path.join(REF, "INDEX.json"), "w") as jf:
        json.dump({"machine": machine, "arch": args.arch,
                   "kernels": summary}, jf, indent=2)
        jf.write("\n")
    print(f"\n{len(summary) - failures}/{len(summary)} kernels produced a "
          f"reference; index in ref/INDEX.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
