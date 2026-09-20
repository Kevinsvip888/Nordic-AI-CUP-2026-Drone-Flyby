# Recorded results

| System | Evidence | Result |
|---|---|---|
| Historical YOLO26-L E40, fixed L0, four-flip top2 fusion | Confirmed official Validation | 0.4950687646541436; 246/249 received frames |
| Final L0 E50 plus three-L0/class-change-first policy | Offline saved-image replay | 248 frames, 198 L0, 50 synthetic L1 |
| Final policy | Complete official Validation | Not available |

Final replay: 5 inspections triggered by class change, 45 by confidence change; 53 memory-only boxes across 31 frames. These are activity counts, not accuracy measures. The replay has no official ground truth. Its L1 input is an interpolated crop of a saved 960x540 image, not a true high-resolution source crop.

Final checkpoint SHA256: `4c13fdfd807fc843257fa58d6bf319cf216c1f96042e570be841ccd43e8493bf`.

Last training stage: YOLO26-L E40 continued for ten epochs using all 47 available L0 training/development images, 120 optimizer updates, 1280 input, AdamW learning rate 1e-5, physical batch 2 and accumulation 2. Earlier held-out local L0 images were explicitly included by the user; that split no longer supplies an independent validation estimate. No claim of meeting the 0.7 target is made.

The included tests cover camera cadence/priority, correction from new observations, motion failure, off-crop memory, exits, Kalman covariance behavior, duplicate requests, out-of-order frames and sequence isolation. They do not prove detection quality or real-time performance on other hardware.
