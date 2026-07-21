"""
patch_room_size.py
==================
Extrae el tamaño real de habitación (ej. "44 m²") de los CSVs de fichas ya
scrapeados, donde actualmente aparece como "room size" en servicios_habitacion.

- Usa requests puro (Playwright solo una vez para resolver el WAF)
- No modifica los CSVs originales
- Genera data/raw/fichas/room_sizes.csv con columnas:
      url_estancia | room_size_raw | room_size_m2
- Es reanudable: si se interrumpe, continúa donde se quedó

Uso:
    python patch_room_size.py
"""

import os, csv, re, time, random, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent))
from Scrp_caracteristicas_estancias import BookingSession

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()
BASE = Path(os.getenv("BASE"))

INPUT_DIR  = BASE / "data" / "raw" / "fichas"
OUTPUT_CSV = INPUT_DIR / "room_sizes.csv"
SEP        = "|"
MAX_WORKERS = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def cargar_urls_desde_fichas() -> list[str]:
    urls = []
    for f in sorted(INPUT_DIR.glob("resultados_booking_*.csv")):
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter=SEP):
                url = row.get("url_estancia", "").strip()
                if url:
                    urls.append(url)
    print(f"  Total URLs en fichas: {len(urls)}")
    return urls


def cargar_ya_procesadas() -> set:
    done = set()
    if not OUTPUT_CSV.exists():
        return done
    with open(OUTPUT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=SEP):
            u = row.get("url_estancia", "").strip()
            if u:
                done.add(u)
    print(f"  Ya procesadas (reanudando): {len(done)}")
    return done


def extraer_room_size(html: str) -> tuple[str, str]:
    """Devuelve (room_size_raw, room_size_m2). room_size_m2 es solo el número."""
    soup = BeautifulSoup(html, "lxml")
    fila = soup.select_one("tr.hprt-table-cheapest-block")
    if not fila:
        return "", ""
    for fac in fila.select(".hprt-facilities-facility"):
        if fac.get("data-name-en", "").lower() == "room size":
            raw = fac.get_text(strip=True)          # "44 m²"
            m   = re.search(r"(\d+(?:[.,]\d+)?)", raw)
            num = m.group(1).replace(",", ".") if m else ""
            return raw, num
    return "", ""

# ── Worker ────────────────────────────────────────────────────────────────────

def procesar_url(url: str, session: BookingSession) -> dict:
    time.sleep(random.uniform(0.5, 1.5))
    r = session.get(url)
    if r is None:
        return {"url_estancia": url, "room_size_raw": "", "room_size_m2": "", "error": "no_response"}
    raw, num = extraer_room_size(r.text)
    return {"url_estancia": url, "room_size_raw": raw, "room_size_m2": num, "error": ""}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  PATCH room_size")
    print("=" * 62)

    todas  = cargar_urls_desde_fichas()
    hechas = cargar_ya_procesadas()
    pendientes = [u for u in todas if u not in hechas]

    print(f"  Pendientes: {len(pendientes)}  |  Workers: {MAX_WORKERS}")
    print("=" * 62)

    if not pendientes:
        print("[OK] Todo ya procesado.")
        return

    print("Iniciando sesión (resolviendo WAF una sola vez)...")
    session = BookingSession()

    modo = "a" if hechas else "w"
    escribir_header = not hechas

    lock          = threading.Lock()
    results_map   = {}
    next_to_write = [0]
    errores       = [0]
    writer        = [None]

    with open(OUTPUT_CSV, modo, newline="", encoding="utf-8-sig") as csv_file:

        def procesar_y_guardar(idx: int, url: str):
            try:
                result = procesar_url(url, session)
            except Exception as e:
                result = {"url_estancia": url, "room_size_raw": "", "room_size_m2": "", "error": str(e)}
                with lock:
                    errores[0] += 1

            with lock:
                results_map[idx] = result
                while next_to_write[0] in results_map:
                    r = results_map.pop(next_to_write[0])
                    if writer[0] is None:
                        writer[0] = csv.DictWriter(
                            csv_file,
                            fieldnames=["url_estancia", "room_size_raw", "room_size_m2", "error"],
                            delimiter=SEP,
                            extrasaction="ignore",
                        )
                        if escribir_header:
                            writer[0].writeheader()
                    writer[0].writerow(r)
                    csv_file.flush()
                    next_to_write[0] += 1

                    # Log cada 100
                    n = next_to_write[0]
                    if n % 100 == 0:
                        pct = n / len(pendientes) * 100
                        print(f"  [{n}/{len(pendientes)}] {pct:.1f}% — {r['url_estancia'][:60]}  m²={r['room_size_m2'] or '–'}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(procesar_y_guardar, i, u): i for i, u in enumerate(pendientes)}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  Future error: {e}")

    print(f"\n{'='*62}")
    print(f"  Completado: {len(pendientes)} procesadas  |  {errores[0]} errores")
    print(f"  Salida: {OUTPUT_CSV}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
