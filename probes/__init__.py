"""
Probe System for Subliminal Learning Research

This package contains utilities for training and using linear probes to detect
traits in language model representations. The probe system is designed to:

1. Extract internal representations from language models at any layer
2. Train linear classifiers to detect specific traits (like animal preferences)
3. Monitor how these traits evolve during finetuning
4. Provide insights into subliminal learning effects

Package Structure:
- core/: Core probe utilities and monitoring classes
- training/: Scripts for training probes and generating data
- analysis/: Analysis and debugging tools
- examples/: Example usage patterns and workflows
- data/: Training data and results
"""