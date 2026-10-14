# PROGRESS.md — Simulator Audit

## How to use this file
If the task is interrupted, read this file FIRST to see where to resume.

---

## 2026-10-13 15:14 — Phase A complete
- Read all existing files on disk.
- Step 1 (understand simulator): COMPLETE — original SIMULATOR/server.py analyzed. Key findings in audit_step3_4_compare.py lines 118-148.
- Step 2 (measure real images): COMPLETE — audit_step2_results.json has all 10 folders, 255 records. Spot-checked and verified.
- Steps 3-7: scripts existed but were never run. No output files.
- Created simulator_v2/ (copy of SIMULATOR/server.py), simulator_audit/ output folder.
- Resume from Step 3.

## 2026-10-13 15:15 — Starting Steps 3-7 execution
- Running comprehensive audit_pipeline.py script that covers Steps 3, 4, 5, 6.
- Outputs go to simulator_audit/per_platform/ (one JSON per platform per step).
- Script is resumable: checks for existing per-platform outputs and skips completed work.

## 2026-10-13 15:18:55 - Step 3: Checking original-to-platform pairing

## 2026-10-13 15:18:55 - Step 3 complete. No filename pairing found. Using distribution matching.

## 2026-10-13 15:18:55 - Step 4: Running simulator comparison (v1 vs real)

## 2026-10-13 15:57:22 - Step 4 complete. Verdicts: WA=TOO HARSH, FB=MOSTLY ACCURATE, TG=MOSTLY ACCURATE, IG=ACCURATE

## 2026-10-13 15:57:22 - Step 6: Validating on held-out test data

## 2026-10-13 16:01:46 - Step 6 validation complete (excluding TruFor downstream check).

## 2026-10-13 16:20:05 - Step 6: TruFor Downstream Check Complete
- Output saved to: D:\ml_project\simulator_audit\per_platform\trufor_downstream_results.json
- WHATSAPP: v1 MAE=0.2515, v2 MAE=0.2515 (v2 closer: True)
- FACEBOOK: v1 MAE=0.1380, v2 MAE=0.1380 (v2 closer: True)
- TELEGRAM: v1 MAE=0.1753, v2 MAE=0.1753 (v2 closer: True)
- INSTAGRAM: v1 MAE=0.0855, v2 MAE=0.0513 (v2 closer: True)

## 2026-10-13 17:26:16 - v3: Starting v3 audit pipeline

## 2026-10-13 17:26:23 - v3: Starting v3 audit pipeline

## 2026-10-13 17:29:36 - v3: Step 1 pairing complete for all platforms

## 2026-10-13 17:31:10 - v3: Step 2 paired BPP complete for all platforms

## 2026-10-13 17:56:34 - v3 Master: Step 1 Pairing complete for all 16 platform-split combinations.

## 2026-10-13 18:02:39 - v3 Master: Step 2 Comparison complete for all platforms.

## 2026-10-13 20:06:50 - v3 Master: Step 1 Pairing complete for all 16 platform-split combinations.

## 2026-10-13 20:06:50 - v3 Master: Step 2 Comparison complete for all platforms.

## 2026-10-13 20:27:30 - v3 Master: Step 3 Complete: Updated v3 server with FB filter=BILINEAR, IG quality=60

## 2026-10-13 20:27:31 - v3 Master: Step 4 Small images analysis complete.

## 2026-10-13 20:30:06 - v3 Master: Step 5 Held-out test validation complete.

## 2026-10-13 20:30:06 - v3 Master: Master pipeline Steps 1-5 finished successfully.

## 2026-10-13 20:57:13 - v3 Step 6 TruFor: Step 6 (i) Paired TruFor check complete.

## 2026-10-13 21:01:46 - v3 Step 6 TruFor: Step 6 (ii) Real-world threshold check complete.

## 2026-10-13 21:01:46 - v3 Step 6 TruFor: TruFor Step 6 downstream evaluation finished successfully.

## 2026-10-13 21:07:20 - All Steps 1-7 Complete: Deliverables (simulator_v3/, AUDIT_REPORT.md, regenerate_compressed_sets_v3.py, README.md) finalized.

## 2026-10-13 22:10:00 - Simulator v3 Gap Closure (Steps 1-4):
- Step 1: Inspected Instagram simulator_v3/server.py. Proved real Instagram upscales images <1440px to width 1440px and center-crops 1:1 by default (25/25 real photos, 20/25 CASIA fakes).
- Step 2: Fit crop rule (1:1 center-crop default), upscale filter (BICUBIC minimizes ringing; noise energy 491 vs real 420; PSNR 33.0-38.1 dB), and WebP quality (q=60 works for both large 0.958x and small 0.947x). Instagram marked ACCURATE. Updated simulator_v3/server.py.
- Step 3: TruFor resolution audit: run_step6_trufor.py used max_size=384; official evaluator uses native size. Reran 25-image paired comparison at max_size=512 for WhatsApp and Instagram. For WhatsApp, MAE remains stable (0.0213). For Instagram, v3 cuts error by 53.1% (MAE 0.0262 vs v1 0.0559) and achieves Pearson r=0.9827 with real Instagram images.
- Step 4: Corrected AUDIT_REPORT.md: (a) Instagram verdict updated to ACCURATE, (b) threshold caveats added (25 real, 10 fake CASIA splices, overlapping CIs, non-generalizability noted), (c) untested CASIA collapse claim removed, (d) tested on 30 CASIA 1.0 originals: v1 AUC=0.8400 vs v3 AUC=0.7289 (diff=-0.1111); upscaling actually degrades small-image splice detection.

## 2026-10-13 22:30:00 - Full Dataset Regeneration Complete (Step 5):
- Executed regenerate_compressed_sets_v3.py across ALL four platforms (whatsapp, facebook, telegram, instagram) and ALL three benchmark datasets (CASIA1.0, CocoGlide, Columbia).
- Output Directory: results/simulated_v3/<dataset>/<platform>/ (never overwriting old sets).
- Verified file counts across all 12 platform-dataset combinations:
  - CASIA1.0: 1721 / 1721 (100% MATCH on WA, FB, TG, IG)
  - CocoGlide: 1024 / 1024 (100% MATCH on WA, FB, TG, IG)
  - Columbia: 363 / 363 (100% MATCH on WA, FB, TG, IG)
  - Total Regenerated Files: 12,432 files in 269.7s.
## 2026-10-14 09:35:00 - Morning Status Check (v3 Baseline Job):
- Verified PID 18508 is actively computing on CPU (~850% multi-threaded utilization, 795 MB RAM).
- Total cached scores: 820 / 1500 (54.7% of test baseline). CocoGlide 100% complete (500/500). Columbia 64% complete (320/500: orig, WA, IG done; FB actively processing).
- Verified major overnight bottleneck (Columbia Instagram at 1440px native, which took 7.9h) is finished.
- Job progressing normally without errors; monitoring until completion before running recalibration analysis.



