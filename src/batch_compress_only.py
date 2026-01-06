import os
import glob
import re
import argparse
import subprocess
import sys

def get_bits_from_path(path):
    """
    Sucht nach 'qbX' im Dateipfad oder Dateinamen, um die Bits zu bestimmen.
    Z.B. 'results/.../qb4_p8_structured_C/...' -> 4 Bits
    """
    # Sucht nach qb gefolgt von Zahlen
    match = re.search(r"qb(\d+)", path)
    if match:
        return int(match.group(1))
    return None

def main(root_dir):
    print(f"--- Starte Batch-Kompression für: {root_dir} ---")
    
    # Suche alle .pth Dateien rekursiv
    search_pattern = os.path.join(root_dir, "**", "*.pth")
    all_pth_files = glob.glob(search_pattern, recursive=True)
    
    tasks = []

    for pth in all_pth_files:
        # 1. Ignoriere bereits komprimierte Dateien oder FHE Circuits
        if "_COMPRESSED.pth" in pth or "_FHE_CIRCUIT.pth" in pth:
            continue
        
        # 2. Ignoriere Dateien, die keine Modelle sind (z.B. wenn .pth.tar o.ä. existiert)
        if not pth.endswith(".pth"):
            continue

        # 3. Prüfe, ob die komprimierte Version schon existiert (um Zeit zu sparen)
        compressed_path = pth.replace(".pth", "_COMPRESSED.pth")
        if os.path.exists(compressed_path):
            # print(f"[SKIP] Bereits komprimiert: {os.path.basename(pth)}")
            continue

        # 4. Bits automatisch erkennen
        bits = get_bits_from_path(pth)
        if bits is None:
            print(f"[WARNUNG] Konnte Bits (qbX) nicht im Pfad finden für: {pth}. Überspringe.")
            continue
            
        tasks.append((pth, bits))

    if not tasks:
        print("Keine neuen Modelle zum Komprimieren gefunden.")
        return

    print(f"Gefunden: {len(tasks)} Modelle, die komprimiert werden müssen.")
    print("-" * 60)

    # Ausführung
    for i, (model_path, bits) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Komprimiere (Bits={bits}): {os.path.basename(model_path)} ...")
        
        # Der Befehl, den du manuell eingegeben hättest
        cmd = [
            sys.executable, "-m", "src.compression.compress_model",
            "--model_path", model_path,
            "--bits", str(bits)
        ]
        
        try:
            # Führe das Modul als Subprozess aus
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"  [FEHLER] Komprimierung fehlgeschlagen (Exit Code {e.returncode})")
        except Exception as e:
            print(f"  [FEHLER] Unerwarteter Fehler: {e}")

    print("-" * 60)
    print("Batch-Kompression abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sucht Modelle in einem Ordner und komprimiert sie automatisch mit den richtigen Bits.")
    parser.add_argument("folder", type=str, help="Pfad zum Meta-Run Ordner (z.B. results/qat_runs/meta_run_...)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"Fehler: Ordner nicht gefunden: {args.folder}")
        sys.exit(1)
        
    main(args.folder)