"""Submission-specific checks; no models, external services or working databases."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_portable_launcher_help_and_invalid_port():
    p=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'--help'],capture_output=True,text=True,timeout=10)
    assert p.returncode==0 and '--check' in p.stdout and '8799' in p.stdout
    p=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'--port','0'],capture_output=True,text=True,timeout=10)
    assert p.returncode==2 and '1024' in p.stderr


def test_formal_solution_stays_within_case_limit():
    text=(ROOT/'submission/solution.md').read_text()
    assert len(text)<=15000
    assert all('## '+str(i)+'.' in text for i in [1,2,3,4,6])
    assert (ROOT/'submission/effect.md').is_file()
