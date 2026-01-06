# In src/training/train_qat_brevitas.py (oder einer neuen utils/config_loader.py)
import yaml
import os
#import sys # Für sys.exit bei kritischen Fehlern

def load_config(config_path="config.yaml"):
    """Lädt eine YAML-Konfigurationsdatei."""
    try:
        # Wenn der config_path nicht absolut ist, versuchen wir ihn relativ zum Skriptpfad
        # oder einem bekannten Projektstamm zu finden.
        # Für jetzt gehen wir davon aus, dass config.yaml im selben Verzeichnis liegt,
        # aus dem das Hauptskript oder Notebook gestartet wird, oder der Pfad ist absolut.
        # In der Setup-Zelle unseres Notebooks ist das Arbeitsverzeichnis der Projektstamm.
        
        # Korrektur, um sicherzustellen, dass der Pfad relativ zum aktuellen Arbeitsverzeichnis ist,
        # wenn kein absoluter Pfad angegeben wird (was in Notebooks oft der Fall ist)
        if not os.path.isabs(config_path):
            # project_root wird in der Setup-Zelle des Notebooks oder am Anfang des Skripts definiert
            # Hier müssen wir sicherstellen, dass er global verfügbar ist oder übergeben wird.
            # Einfacher ist es, wenn config.yaml im CWD liegt.
             pass # Gehe davon aus, config_path ist relativ zum CWD oder absolut

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Falls die Datei leer ist
            print(f"FEHLER: Konfigurationsdatei {config_path} ist leer oder konnte nicht geparst werden.")
            return {} # Leeres Dict zurückgeben oder Fehler werfen
        print(f"Konfiguration erfolgreich von {config_path} geladen.")
        return config
    except FileNotFoundError:
        print(f"FEHLER: Konfigurationsdatei nicht gefunden unter {config_path}")
        # sys.exit(1) # Beendet das Skript, wenn die Config kritisch ist
        return {} # Leeres Dict zurückgeben oder Fehler werfen
    except yaml.YAMLError as e:
        print(f"FEHLER: Beim Parsen der YAML-Konfigurationsdatei {config_path}: {e}")
        # sys.exit(1)
        return {}
    except Exception as e:
        print(f"FEHLER: Ein unerwarteter Fehler ist beim Laden der Konfigurationsdatei aufgetreten: {e}")
        return {}