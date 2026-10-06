---
name: where-things-go
description: On the Strix Halo PC, conda envs and model-sized data live on C: (NVMe); Strata source, builds and results stay in the repo on E: (USB HDD)
metadata:
  type: feedback
---

Layout rule for the EVO-X2 (Strix Halo) PC: conda environments and everything model-sized (GGUFs, packs, the MTP
draft layer) go on C:, the NVMe. The Strata source, its build directories (`build-hip-win`, `dist\`, `engine\`) and
results (logs, benchmark JSON, docs) stay in the repository at E:\work\AI\Strata, even though E: is a slower exFAT
USB hard disk.

**Why:** the user corrected my "BUILD_DIR on C:" suggestion on 2026-10-05: "conda env和模型在c盘较快，strata代码build和结果存在本项目路径".

**How to apply:** never move a build dir or results off the repo for speed; never put a conda env or model files on E:. Related: [[conda-env-policy]].
