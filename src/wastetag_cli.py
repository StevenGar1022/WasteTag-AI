"""WASTETAG-AI — Herramienta interactiva de Auto-Etiquetado (Fase 4.5).

Comando global (tras `pip install -e .` dentro de env/): `wastetag-ai`
- Sin flags -> sesión interactiva paso a paso.
- Con flags (--source ...) -> modo directo, sin preguntas.

Ejemplos:
    wastetag-ai
    wastetag-ai --source data/samples --weights runs/train/yolo26s_autos/weights/best.pt
    wastetag-ai --source IMG.jpg --conf 0.5 --no-abrir
"""
import argparse
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auto_label import collect_images, etiquetar_imagen  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import Progress, BarColumn, TextColumn  # noqa: E402
from rich.prompt import Prompt, FloatPrompt, IntPrompt  # noqa: E402
from rich.table import Table  # noqa: E402
from rich import box  # noqa: E402

console = Console()
ROOT = Path(__file__).parent.parent


def banner() -> None:
    console.print(Panel(
        "[bold cyan]WASTETAG-AI[/bold cyan]\n"
        "[white]Herramienta interactiva de Auto-Etiquetado[/white]",
        box=box.DOUBLE, expand=False,
    ))


def buscar_modelos() -> list[Path]:
    cands = sorted(ROOT.glob("runs/**/weights/*.pt"))
    cands += sorted((ROOT / "weights").glob("*.pt"))
    cands += [ROOT / "yolo26s.pt"] if (ROOT / "yolo26s.pt").exists() else []
    seen, out = set(), []
    for p in cands:
        r = p.resolve()
        if r not in seen and r.is_file():
            seen.add(r)
            out.append(p)
    return out


def elegir_modelo(weights_arg: str | None, interactivo: bool) -> Path:
    if weights_arg:
        p = Path(weights_arg)
        if not p.is_file():
            console.print(f"[red]No existe: {p}[/red]")
            raise SystemExit(1)
        return p
    default = ROOT / "runs/train/yolo26s_autos/weights/best.pt"
    if not interactivo:
        if default.is_file():
            return default
        console.print("[red]Sin --weights y sin best.pt por defecto.[/red]")
        raise SystemExit(1)
    modelos = buscar_modelos()
    console.print("\n[bold]1. Mejor modelo entrenado (.pt)[/bold]")
    if modelos:
        t = Table(box=box.SIMPLE)
        t.add_column("#", justify="right")
        t.add_column("Modelo")
        t.add_column("Tamaño", justify="right")
        t.add_column("Modificado")
        for i, m in enumerate(modelos, 1):
            st = m.stat()
            t.add_row(str(i), str(m),
                      f"{st.st_size / 1e6:.1f} MB",
                      datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"))
        t.add_row("0", "[italic]Otra ruta...[/italic]", "", "")
        console.print(t)
        op = IntPrompt.ask("Elige", default=1)
        if op == 0:
            ruta = Prompt.ask("Ruta al .pt")
        else:
            if not (1 <= op <= len(modelos)):
                console.print("[red]Opción inválida[/red]")
                raise SystemExit(1)
            return modelos[op - 1]
    else:
        console.print("[yellow]No se encontró ningún .pt en runs/ ni weights/.[/yellow]")
        ruta = Prompt.ask("Ruta al .pt")
    p = Path(ruta)
    if not p.is_file():
        console.print(f"[red]No existe: {p}[/red]")
        raise SystemExit(1)
    return p


def pedir_imagenes(source_arg: str | None, interactivo: bool) -> list[Path]:
    console.print("\n[bold]2. Imágenes a etiquetar[/bold]")
    if not interactivo:
        p = Path(source_arg)
        imgs = collect_images(p) if p.exists() else []
        if not imgs:
            console.print(f"[red]Sin imágenes válidas en: {p}[/red]")
            raise SystemExit(1)
        console.print(f"Imágenes totales: [bold]{len(imgs)}[/bold]")
        return imgs
    s = None
    while True:
        s = Prompt.ask("Carpeta o imagen")
        p = Path(s)
        if not p.exists():
            console.print(f"[red]No existe: {p}[/red]")
            continue
        imgs = collect_images(p)
        if not imgs:
            console.print("[red]Sin imágenes válidas (.jpg/.jpeg/.png/.bmp/.webp)[/red]")
            continue
        console.print(f"Imágenes totales: [bold]{len(imgs)}[/bold]")
        return imgs


def pedir_params(args, interactivo: bool) -> dict:
    console.print("\n[bold]3. Parámetros de etiquetado[/bold] [dim](Enter = aceptar)[/dim]")
    if args.conf is None:
        args.conf = (FloatPrompt.ask("Nivel de confianza", default=0.35)
                     if interactivo else 0.35)
    if args.imgsz is None:
        args.imgsz = (IntPrompt.ask("Resolución (imgsz)", default=640)
                      if interactivo else 640)
    if args.device is None:
        args.device = (Prompt.ask("Dispositivo (0 / cpu)", default="0")
                       if interactivo else "0")
    if args.output is None:
        args.output = (Prompt.ask("Carpeta salida labels", default="auto_labels")
                       if interactivo else "auto_labels")
    console.print(f"conf={args.conf} imgsz={args.imgsz} device={args.device} "
                  f"output={args.output}")
    return args


def cargar_modelo(weights: Path, device) -> object:
    from ultralytics import YOLO
    with console.status("[bold green]CARGANDO MODELO...[/bold green]"):
        model = YOLO(str(weights))
    try:
        names = model.names
        console.print(f"Modelo OK — clases: {[names[i] for i in sorted(names)]}")
    except Exception:
        names = {}
    return model


def contact_sheet(model, names: dict, elegidas: list[Path], out_dir: Path,
                  conf: float, imgsz: int, device) -> Path | None:
    """Re-ejecuta N imágenes, dibuja cajas y arma mosaico de verificación."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception as e:
        console.print(f"[yellow]Sin PIL/numpy para mosaico: {e}[/yellow]")
        return None
    tiles = []
    for img_path in elegidas:
        r = model.predict(source=str(img_path), conf=conf, imgsz=imgsz,
                          device=device, verbose=False, save=False)[0]
        anot = Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB
        anot.thumbnail((640, 640))
        d = ImageDraw.Draw(anot)
        d.text((8, 8), img_path.name, fill=(255, 255, 0))
        tiles.append(anot)
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    dest = out_dir / "contact_sheet_verificacion.jpg"
    sheet.save(dest)
    return dest


def abrir_visor(path: Path) -> None:
    try:
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import os
            os.startfile(str(path))  # noqa: S606
        console.print(f"[green]Mosaico abierto en el visor.[/green]")
    except Exception:
        console.print("[yellow]No se pudo abrir el visor automáticamente.[/yellow]")
    console.print(f"Verificación visual: [bold]{path.resolve()}[/bold]")


def main() -> int:
    ap = argparse.ArgumentParser(prog="wastetag-ai",
                                 description="WASTETAG-AI — Herramienta interactiva de Auto-Etiquetado")
    ap.add_argument("--source", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--racha-alerta", type=int, default=200,
                    help="Racha de no-etiquetadas que dispara diagnóstico (default 200)")
    ap.add_argument("--muestras", type=int, default=4,
                    help="Nº imágenes aleatorias para verificación visual")
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--no-abrir", action="store_true", help="No abrir el visor")
    args = ap.parse_args()

    banner()
    interactivo = args.source is None
    weights = elegir_modelo(args.weights, interactivo)
    imagenes = pedir_imagenes(args.source, interactivo)
    args = pedir_params(args, interactivo)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = cargar_modelo(weights, args.device)
    names = getattr(model, "names", {}) or {}

    total = len(imagenes)
    with_det, without_det, total_boxes = 0, 0, 0
    conf_sum, conf_n = 0.0, 0
    por_clase: dict[int, int] = {}
    racha, rachas = 0, []  # rachas: [(img_inicio, largo)]
    racha_inicio = None
    etiquetadas: list[Path] = []

    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  console=console) as prog:
        task = prog.add_task("Etiquetando", total=total)
        for img_path in imagenes:
            prog.update(task, description=f"[cyan]{img_path.name}[/cyan]")
            try:
                r = etiquetar_imagen(model, img_path, out_dir,
                                     args.conf, args.imgsz, args.device)
            except Exception as e:
                without_det += 1
                racha += 1
                if racha == 1:
                    racha_inicio = img_path.name
                prog.advance(task)
                console.print(f"[red]ERROR {img_path.name}: {e}[/red]")
                continue
            if r["saved"] == 0:
                without_det += 1
                racha += 1
                if racha == 1:
                    racha_inicio = img_path.name
                if racha == args.racha_alerta:
                    rachas.append((racha_inicio, racha))
            else:
                with_det += 1
                etiquetadas.append(img_path)
                total_boxes += r["saved"]
                racha = 0
                racha_inicio = None
                for b in r["boxes"]:
                    por_clase[b["cls"]] = por_clase.get(b["cls"], 0) + 1
                    conf_sum += b["conf"]
                    conf_n += 1
            prog.update(task, advance=1,
                        description=f"✓{with_det} ✗{without_det} 📦{total_boxes}")
    if racha >= args.racha_alerta:
        rachas.append((racha_inicio, racha))

    # ---- Resumen de la sesión ----
    console.print(Panel("[bold]RESUMEN DE LA SESIÓN[/bold]", expand=False))
    t = Table(box=box.SIMPLE)
    t.add_column("Métrica")
    t.add_column("Valor", justify="right")
    cov = 100 * with_det / total if total else 0
    t.add_row("Imágenes totales", str(total))
    t.add_row("Etiquetadas (txt YOLO)", f"{with_det} ({cov:.1f}%)")
    t.add_row("No etiquetadas", str(without_det))
    t.add_row("Cajas generadas", str(total_boxes))
    t.add_row("Cajas / imagen", f"{total_boxes / with_det:.2f}" if with_det else "-")
    t.add_row("Confianza media", f"{conf_sum / conf_n:.3f}" if conf_n else "-")
    console.print(t)
    if por_clase:
        tc = Table(title="Cajas por clase", box=box.SIMPLE)
        tc.add_column("ID", justify="right")
        tc.add_column("Clase")
        tc.add_column("Cajas", justify="right")
        for cid in sorted(por_clase):
            tc.add_row(str(cid), str(names.get(cid, "?")), str(por_clase[cid]))
        console.print(tc)

    # ---- Diagnóstico ----
    if with_det == 0:
        ta = Table(title="⚠ ALERTA DE RENDIMIENTO", box=box.HEAVY, style="red")
        ta.add_column("Síntoma")
        ta.add_column("Causa probable")
        ta.add_row("Ninguna imagen etiquetada",
                   "conf muy alto, modelo de otro dominio, pesos o imgsz inadecuados")
        console.print(ta)
        console.print("[red]Revisa conf, modelo y dominio antes de reintentar.[/red]")
    elif rachas:
        ta = Table(title="⚠ ALERTA DE RENDIMIENTO", box=box.HEAVY, style="yellow")
        ta.add_column("Racha sin etiquetar")
        ta.add_column("Inicia en")
        for ini, largo in rachas:
            ta.add_row(f"{largo} imágenes", ini)
        console.print(ta)
        console.print("[yellow]Diagnóstico: el modelo base etiquetó y luego dejó de hacerlo — "
                      "es muy probable que exista un sobreajuste (overfitting) o un "
                      "desequilibrio de pesos entre las clases.[/yellow]")
    else:
        console.print(Panel("[bold green]AUTO ETIQUETADO EXITOSO[/bold green] "
                            "— todas las imágenes procesadas con detecciones hasta la última.",
                            expand=False))

    # ---- Validación visual aleatoria ----
    if etiquetadas and args.muestras > 0:
        rnd = random.Random(args.semilla)
        elegidas = rnd.sample(etiquetadas, min(args.muestras, len(etiquetadas)))
        console.print(f"\n[bold]Verificación visual[/bold] ({len(elegidas)} al azar): "
                      + ", ".join(p.name for p in elegidas))
        sheet = contact_sheet(model, names, elegidas, out_dir,
                              args.conf, args.imgsz, args.device)
        if sheet:
            if not args.no_abrir:
                abrir_visor(sheet)
            else:
                console.print(f"Mosaico: [bold]{sheet.resolve()}[/bold]")
            console.print("Mira si etiquetó bien o si el modelo necesita revisión "
                          "de pesos de clases / sobreajuste.")
    console.print(f"\nLabels en: [bold]{out_dir.resolve()}[/bold]")
    return 0 if with_det else 2


if __name__ == "__main__":
    raise SystemExit(main())
