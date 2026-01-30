import os
import re
import csv
import argparse

def extract_peak_ram(file_path):
    """
    Öffnet eine Datei und sucht nach dem Peak RAM Wert mittels Regex.

    Args:
        file_path (str): Der vollständige Pfad zur Textdatei.

    Returns:
        str: Der gefundene Wert inklusive Einheit (z.B. "123.45 MB") oder "N/A",
             falls nichts gefunden wurde.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Suchmuster für Peak RAM. 
            # Es sucht nach "Peak RAM", ignoriert Groß/Kleinschreibung,
            # erlaubt beliebige Zeichen bis zur Zahl und fängt Zahl + Einheit (MB/GB) ein.
            # Beispiel matcht: "Peak RAM: 500 MB" oder "Peak RAM usage = 12.5 GB"
            match = re.search(r"Peak RAM.*?(\d+(?:\.\d+)?)\s*([M|G|K]B)", content, re.IGNORECASE)
            
            if match:
                return f"{match.group(1)} {match.group(2)}"
            else:
                return "N/A"
    except Exception as e:
        print(f"Fehler beim Lesen von {file_path}: {e}")
        return "Error"


def scan_and_compile_csv(meta_folder_path, output_csv_path):
    """
    Durchsucht den Ordner rekursiv nach *_COMPRESSED_execute.txt Dateien,
    extrahiert RAM-Werte und speichert sie in einer CSV.

    Args:
        meta_folder_path (str): Pfad zum Hauptordner (Meta-Ordner).
        output_csv_path (str): Pfad, wo die CSV gespeichert werden soll.
    """
    data_rows = []
    
    # Header für die CSV-Datei
    header = ['Model Path', 'Filename', 'Peak RAM']

    print(f"Starte Suche in: {meta_folder_path}...")

    # os.walk durchläuft den Verzeichnisbaum rekursiv
    for root, dirs, files in os.walk(meta_folder_path):
        for filename in files:
            # Filterung auf die gewünschte Dateiendung
            if filename.endswith("_COMPRESSED_execute.txt"):
                full_path = os.path.join(root, filename)
                
                # Extrahiere den RAM Wert
                ram_value = extract_peak_ram(full_path)
                
                # Relativen Pfad zum Model ermitteln (für sauberere CSV)
                # Falls der absolute Pfad gewünscht ist, einfach 'root' nutzen.
                model_path = os.path.relpath(root, meta_folder_path)
                
                data_rows.append([model_path, filename, ram_value])

    # Schreiben der CSV-Datei
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data_rows)
        
        print(f"\nErfolg! {len(data_rows)} Einträge wurden in '{output_csv_path}' gespeichert.")
        
    except IOError as e:
        print(f"Fehler beim Schreiben der CSV-Datei: {e}")


if __name__ == "__main__":
    # Argument Parser für die Kommandozeilennutzung
    parser = argparse.ArgumentParser(description="Extrahiert Peak RAM aus execute.txt Dateien.")
    
    parser.add_argument("folder", help="Der Pfad zum Meta-Ordner, der durchsucht werden soll.")
    parser.add_argument("--out", default="peak_ram_report.csv", help="Name der Ausgabedatei (Default: peak_ram_report.csv)")

    args = parser.parse_args()

    if os.path.isdir(args.folder):
        scan_and_compile_csv(args.folder, args.out)
    else:
        print(f"Der angegebene Pfad existiert nicht: {args.folder}")