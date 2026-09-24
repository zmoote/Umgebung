# GEMINI.md: Umgebung Development & Architecture Guide

## 1. Project Overview & Core Philosophy

**Umgebung** is an interactive reality simulation engine as informed and described by `Thinkers` in the research submodule, including but not limited to:
- Elena Danaan
- Alex Collier
- Nassim Haramein
- Chris Essonne
- Christopher Cooper
- Dan Willis
- Marcel Vogel
- Salvatore Pais
- Masaru Emoto

---

## 2. Target Hardware Topology Matrix

The application automatically fetches system configurations at runtime to adapt to host hardware capabilities.

| Target System | CPU Specs | GPU / Accelerator Specs | Execution Strategy |
| --------------| --- | --- | --- |
| **Desktop PC** | AMD Ryzen 9 3900XT (12c/24t) | NVIDIA TITAN RTX (24 GB VRAM) | **Local CUDA Engine:** Large continuous grid allocations directly in GPU VRAM. |
| **Laptop PC** | Intel Core i9-13900H (20t) | NVIDIA RTX 4070 Laptop (8 GB VRAM) | **Local CUDA Engine:** Auto-tuned streamed tile chunking to prevent VRAM overflow. |
| **UMaine ARCSIM** | AMD EPYC Nodes | NVIDIA DGX A100 / RTX Multi-GPU | **Headless CUDA Engine:** Multi-GPU scaling via NVLink and SLURM batch jobs. |
| **UNH Premise** | AMD EPYC / Intel Xeon | NVIDIA A100 / V100 GPUs | **Headless CUDA Engine:** Automated VRAM discovery and stream graph. |
|**RIT SPORC**| Intel (x86_64) | 100x A100 | 