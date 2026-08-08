# Physical-width CPU performance

This is a release-mode measurement of the existing CPU pipeline using
`test_data/raw_sample.NEF`. The decoded/rendered frame is 6000×4000 (24 MP),
Portra 400, seed 1, halation enabled, and `PositiveLinear` output. The width
override is the only setting changed between runs.

| physical width | pitch at 6000 px | pixel-pipeline CPU time | wall time | peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 35 mm | 6.000 µm/px | 7.20 s | 7.83 s | 3.28 GiB |
| 1 mm | 0.1667 µm/px | 69.56 s | 70.06 s | 4.17 GiB |
| 0.5 mm | 0.0833 µm/px | 142.91 s | 143.38 s | not retained |

The measurements were run with the release binary and `/usr/bin/time -l`.
The 0.5 mm timing completed during artifact generation; its later peak-RSS
capture was interrupted, so no memory value is inferred.

Portra's 70 µm halation PSF therefore grows from a 35-pixel Gaussian radius at
35 mm to approximately 1260 pixels at 1 mm and 2520 pixels at 0.5 mm (before
the multi-bounce √3 expansion). The runtime increase is the physical
wide-kernel and grain cost, not a reduced-kernel or display-texture fallback.
No physical kernel was silently shrunk.

The benchmark reports whole-pipeline time; blur and grain are currently
included in that total rather than separately instrumented. Output stability
and fixed-seed determinism are covered by the CPU width tests.
