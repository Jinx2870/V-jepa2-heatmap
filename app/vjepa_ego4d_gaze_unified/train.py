"""
Compatibility shim so we can launch TF+AR gaze training via `app/main.py`.

Launcher flow:
  python app/main.py --fname <yaml>
  -> app/scaffold.py imports `app.<app>.train`
  -> which calls `main(args=...)` below
"""

from __future__ import annotations

from app.vjepa_ego4d.train_gaze_unified import main as _main


def main(args, resume_preempt: bool = False):
    # `resume_preempt` is ignored in this minimal trainer.
    return _main(args)


