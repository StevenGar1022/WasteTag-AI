"""WASTETAG-AI — instalador multiplataforma (Windows / Linux / macOS).
Uso:  python install.py   (Windows:  py install.py)
Crea env/, instala requirements.txt y registra el comando wastetag-ai.
NOTA: se llama install.py (y no setup.py) para no chocar con setuptools.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ENV = ROOT / "env"
WIN = sys.platform == "win32"
PY = str(ENV / ("Scripts/python.exe" if WIN else "bin/python"))


def run(*cmd: str) -> None:
    print("+", " ".join(cmd))
    subprocess.run(list(cmd), check=True, cwd=ROOT)


def main() -> int:
    print("=== WASTETAG-AI setup ===")
    if not Path(PY).is_file():
        print("[1/3] Creando entorno virtual env/...")
        run(sys.executable, "-m", "venv", str(ENV))
    else:
        print("[1/3] env/ ya existe, se reutiliza.")
    print("[2/3] Instalando dependencias...")
    run(PY, "-m", "pip", "install", "-r", "requirements.txt")
    print("[3/3] Registrando comando wastetag-ai...")
    run(PY, "-m", "pip", "install", "-e", ".")
    print("\nOK. Ejecuta el CLI:")
    print("  Linux/macOS:  ./wastetag-ai")
    print("  Windows:      wastetag-ai.bat")
    run(PY, "-m", "src.wastetag_cli", "--help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
