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
from auto_label import collect_images, etiquetar_imagen, etiquetar_lote  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import Progress, BarColumn, TextColumn  # noqa: E402
from rich.prompt import Prompt, FloatPrompt, IntPrompt, Confirm  # noqa: E402
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
    if not modelos:
        console.print("[yellow]No se encontró ningún .pt en runs/ ni weights/.[/yellow]")
        console.print("[dim]Puedes usar tu propio .pt, o escribir 'yolo26s.pt' para descargar la base.[/dim]")
    while True:
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
                ruta = Prompt.ask("Ruta a tu .pt")
            elif 1 <= op <= len(modelos):
                return modelos[op - 1]
            else:
                console.print("[red]Opción inválida, intenta de nuevo.[/red]")
                continue
        else:
            ruta = Prompt.ask("Ruta a tu .pt")
        p = Path(ruta)
        if p.is_file() and p.suffix.lower() == ".pt":
            return p
        if p.name.startswith("yolo") and p.suffix.lower() == ".pt":
            return p  # YOLO() lo descarga solo si falta
        console.print(f"[red]No es un .pt válido: {p}. Intenta de nuevo.[/red]")


IMGSZ_VALIDOS = (320, 480, 640, 960, 1280)


def pedir_imagenes(source_arg: str | None, interactivo: bool,
                   recursivo: bool) -> list[Path]:
    console.print("\n[bold]2. Imágenes a etiquetar[/bold]")
    if not interactivo:
        p = Path(source_arg)
        imgs = collect_images(p, recursivo) if p.exists() else []
        if not imgs:
            console.print(f"[red]Sin imágenes válidas en: {p}[/red]")
            raise SystemExit(1)
        console.print(f"Imágenes totales: [bold]{len(imgs)}[/bold]")
        return imgs
    while True:
        s = Prompt.ask("Carpeta o imagen")
        p = Path(s)
        if not p.exists():
            console.print(f"[red]No existe: {p}[/red]")
            continue
        imgs = collect_images(p, recursivo)
        if not imgs:
            console.print("[red]Sin imágenes válidas (.jpg/.jpeg/.png/.bmp/.webp)[/red]")
            continue
        console.print(f"Imágenes totales: [bold]{len(imgs)}[/bold]")
        return imgs


def pedir_params(args, interactivo: bool) -> dict:
    console.print("\n[bold]3. Parámetros de etiquetado[/bold] [dim](Enter = aceptar)[/dim]")
    if args.conf is None:
        if interactivo:
            while True:
                v = FloatPrompt.ask("Nivel de confianza", default=0.35)
                if 0.0 < v < 1.0:
                    args.conf = v
                    break
                console.print("[red]Confianza debe estar entre 0 y 1.[/red]")
        else:
            args.conf = 0.35
    if not (0.0 < args.conf < 1.0):
        console.print(f"[red]--conf inválido ({args.conf}), usa valor entre 0 y 1.[/red]")
        raise SystemExit(1)
    if args.imgsz is None:
        if interactivo:
            while True:
                v = IntPrompt.ask(f"Resolución {list(IMGSZ_VALIDOS)}", default=640)
                if v in IMGSZ_VALIDOS:
                    args.imgsz = v
                    break
                console.print(f"[red]Elige uno de {list(IMGSZ_VALIDOS)}.[/red]")
        else:
            args.imgsz = 640
    if args.imgsz not in IMGSZ_VALIDOS:
        console.print(f"[red]--imgsz inválido ({args.imgsz}).[/red]")
        raise SystemExit(1)
    if args.device is None:
        args.device = (Prompt.ask("Dispositivo (0 / cpu)", default="0")
                       if interactivo else "0")
    if not str(args.device).strip():
        console.print("[red]Dispositivo vacío.[/red]")
        raise SystemExit(1)
    if args.output is None:
        args.output = (Prompt.ask("Carpeta salida labels", default="auto_labels")
                       if interactivo else "auto_labels")
    console.print(f"conf={args.conf} imgsz={args.imgsz} device={args.device} "
                  f"output={args.output} lote={args.lote} "
                  f"recursivo={args.recursivo} sobrescribir={args.sobrescribir}")
    return args


def acumular(r: dict, img_path: Path, st: dict) -> None:
    """Actualiza contadores con el resultado de una imagen."""
    if r["saved"] == 0:
        st["without"] += 1
        st["racha"] += 1
        if st["racha"] == 1:
            st["racha_ini"] = img_path.name
        if st["racha"] == st["alerta"]:
            st["rachas"].append((st["racha_ini"], st["racha"]))
    else:
        st["with"] += 1
        st["etiquetadas"].append(img_path)
        st["cajas"] += r["saved"]
        st["racha"] = 0
        st["racha_ini"] = None
        for b in r["boxes"]:
            st["clases"][b["cls"]] = st["clases"].get(b["cls"], 0) + 1
            st["conf_sum"] += b["conf"]
            st["conf_n"] += 1


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
    ap.add_argument("-r", "--recursivo", action="store_true",
                    help="Buscar imágenes también en subcarpetas")
    ap.add_argument("--lote", type=int, default=32,
                    help="Tamaño de lote de inferencia (rápido para miles de imgs)")
    ap.add_argument("--sobrescribir", action="store_true",
                    help="Re-etiquetar aunque ya exista el .txt")
    args = ap.parse_args()

    banner()
    interactivo = args.source is None
    weights = elegir_modelo(args.weights, interactivo)
    if interactivo and not args.recursivo:
        args.recursivo = Confirm.ask("¿Buscar en subcarpetas?", default=False)
    imagenes = pedir_imagenes(args.source, interactivo, args.recursivo)
    args = pedir_params(args, interactivo)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = cargar_modelo(weights, args.device)
    names = getattr(model, "names", {}) or {}

    total = len(imagenes)
    st = {"with": 0, "without": 0, "omitidas": 0, "cajas": 0,
          "conf_sum": 0.0, "conf_n": 0, "clases": {},
          "racha": 0, "rachas": [], "racha_ini": None,
          "alerta": args.racha_alerta, "etiquetadas": []}

    def ya_existe(p: Path) -> bool:
        return (out_dir / (p.stem + ".txt")).is_file()

    pendientes = [p for p in imagenes
                  if args.sobrescribir or not ya_existe(p)]
    st["omitidas"] = total - len(pendientes)
    if st["omitidas"]:
        console.print(f"[dim]Se omiten {st['omitidas']} ya etiquetadas "
                      f"(usa --sobrescribir para rehacerlas).[/dim]")

    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  console=console) as prog:
        task = prog.add_task("Etiquetando", total=total)
        if st["omitidas"]:
            prog.advance(task, st["omitidas"])
        lote = max(1, args.lote)
        for i in range(0, len(pendientes), lote):
            chunk = pendientes[i:i + lote]
            try:
                res = (etiquetar_lote(model, chunk, out_dir, args.conf,
                                      args.imgsz, args.device) if len(chunk) > 1
                       else [etiquetar_imagen(model, chunk[0], out_dir,
                                              args.conf, args.imgsz, args.device)])
            except Exception as e:
                for img_path in chunk:
                    st["without"] += 1
                    st["racha"] += 1
                    if st["racha"] == 1:
                        st["racha_ini"] = img_path.name
                    prog.advance(task)
                console.print(f"[red]ERROR lote {chunk[0].name} (+{len(chunk)-1}): {e}[/red]")
                continue
            for img_path, r in zip(chunk, res):
                acumular(r, img_path, st)
                prog.update(task, advance=1,
                            description=f"✓{st['with']} ✗{st['without']} 📦{st['cajas']}")
    if st["racha"] >= args.racha_alerta:
        st["rachas"].append((st["racha_ini"], st["racha"]))
    with_det, without_det, total_boxes = st["with"], st["without"], st["cajas"]
    conf_sum, conf_n, por_clase = st["conf_sum"], st["conf_n"], st["clases"]
    etiquetadas, rachas = st["etiquetadas"], st["rachas"]

    # ---- Resumen de la sesión ----
    console.print(Panel("[bold]RESUMEN DE LA SESIÓN[/bold]", expand=False))
    t = Table(box=box.SIMPLE)
    t.add_column("Métrica")
    t.add_column("Valor", justify="right")
    cov = 100 * with_det / total if total else 0
    t.add_row("Imágenes totales", str(total))
    t.add_row("Etiquetadas (txt YOLO)", f"{with_det} ({cov:.1f}%)")
    t.add_row("No etiquetadas", str(without_det))
    if st["omitidas"]:
        t.add_row("Omitidas (ya existían)", str(st["omitidas"]))
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
    if st["omitidas"] == total:
        console.print(Panel("[bold green]NADA NUEVO POR ETIQUETAR[/bold green] — "
                            "todas las imágenes ya tienen su .txt "
                            "(usa --sobrescribir para rehacerlas).",
                            expand=False))
    elif with_det == 0:
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
