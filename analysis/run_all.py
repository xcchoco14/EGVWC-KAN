"""Runs the analysis scripts in order. Outputs go to analysis/output (JSON, CSV) and analysis/figures."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for s in ("s1_v1_preregistered_test", "s2_v1_benchmark", "s3_interpretability", "s4_stage1", "s5_stage2", "s6_diagnosis",
          "s7_discussion_checks", "s8_robustness", "s9_figures"):
    print(f"\n== {s}")
    subprocess.run([sys.executable, os.path.join(HERE, s + ".py")], check=True, cwd=HERE)
