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
Here I list my two personal machines, followed by the High-Performance Computing (HPC) resources at various universities, which I plan to apply to for a Doctorate Physics or Astrophysics Ph.D. Program.
As of October 3rd, 2026, I am waiting to hear back from UMaine about their decision for Spring 2027; The other universities would be Fall 2027.

Systems will require an NVIDIA graphics card for CUDA acceleration.

| Target System | CPU | GPU | RAM |
| --------------| --- | --- | --- |
| **Personal Desktop** | AMD Ryzen 9 3900XT (12c/24t) | NVIDIA TITAN RTX (24 GB VRAM) | 32 GB DDR4 |
| **Personal Laptop** | Intel Core i9-13900H (14c (6 Performance cores, 8 Efficient cores)/20t) | NVIDIA RTX 4070 Laptop (8 GB VRAM) | 32 GB DDR5 |
| **UMaine ARCSIM** | AMD EPYC Nodes | NVIDIA DGX A100 / RTX Multi-GPU | &ge; 512GB |
| **UNH Premise** | AMD EPYC / Intel Xeon | NVIDIA A100 / V100 GPUs | &ge; 128GB |
|**RIT SPORC**| Intel (x86_64) | 100x NVIDIA A100 GPUs | 24TB |