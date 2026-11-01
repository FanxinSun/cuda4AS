# M2 evidence gate field contract

The historical M2 return schema used `candidate_gate.complete_pass` for the
five integrity and patch-binding events. That value means only that the
integrity sub-gate passed; it never meant that the candidate source build
passed. The field remains in analysis records for compatibility with existing
readers.

New analysis records also carry:

* `candidate_gate.integrity_pass`: the unambiguous name for the historical
  integrity value;
* `candidate_gate.build_outcome`: `PASS`, `FAIL`, `NOT_RUN_ENVIRONMENT`, or
  `NOT_RECORDED`, derived from the `_candidate/build` event when available;
* `candidate_gate.build_pass`: true only when `build_outcome` is `PASS`.

Consequently the accepted M2 retry is represented as
`integrity_pass=true`, `complete_pass=true`, `build_outcome=FAIL`, and
`build_pass=false`. The three required cases remain `NOT_RUN`. A reader must
use `build_outcome` or `build_pass` for source-build decisions and may use
`complete_pass` only for historical integrity compatibility.
