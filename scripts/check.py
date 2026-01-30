import os
import glob
import re

# Pfad zu deinem Meta-Run
SEARCH_DIR = "results/qat_runs/meta_run_20251206_C_q357"

def check_failed_logs():
    print(f"--- DIAGNOSE START: Suche Logs in {SEARCH_DIR} ---\n")
    
    # Wir suchen nach den neuen _cluster_execute.txt Dateien
    files = glob.glob(os.path.join(SEARCH_DIR, "**", "*_cluster_execute.txt"), recursive=True)
    
    if not files:
        print("FEHLER: Keine '*_cluster_execute.txt' Dateien gefunden.")
        return

    print(f"{len(files)} Log-Dateien gefunden. Prüfe die ersten 3...\n")
    
    for fpath in files[:3]:
        print(f"DATEI: {os.path.basename(fpath)}")
        size = os.path.getsize(fpath)
        print(f"GRÖSSE: {size} Bytes")
        
        if size == 0:
            print("STATUS: [LEER] (Die Datei ist komplett leer!)")
            print("-" * 60)
            continue
            
        try:
            with open(fpath, 'r', errors='replace') as f:
                content = f.read()
                
            # 1. Zeige die letzten 500 Zeichen (wo die Ergebnisse stehen sollten)
            print("--- ENDE DER DATEI (RAW) ---")
            print(repr(content[-500:])) # repr() zeigt \n, \t und Farbcodes an!
            print("----------------------------")
            
            # 2. Teste die Regex direkt hier
            print("--- REGEX TEST ---")
            
            # ANSI bereinigen (wie im Hauptskript)
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_content = ansi_escape.sub('', content)
            
            # Teste Zeit-Regex
            time_pat = r"Sample\s+\d+:\s+([0-9\.]+)\s*s"
            times = re.findall(time_pat, clean_content)
            print(f"Gefundene Sample-Zeiten: {times}")
            
            # Teste Fallback-Regex
            fallback_pat = r"Time per sample.*:\s+([0-9\.]+)"
            fallback = re.search(fallback_pat, clean_content)
            print(f"Fallback Zeit Match: {fallback.group(1) if fallback else 'NICHTS'}")

        except Exception as e:
            print(f"FEHLER beim Lesen: {e}")
            
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    check_failed_logs()