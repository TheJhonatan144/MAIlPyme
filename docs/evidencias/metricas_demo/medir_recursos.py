import psutil
import time
import csv
from datetime import datetime

ARCHIVO = "recursos_demo.csv"

print("Midiendo CPU y RAM del sistema. Presiona Ctrl+C para detener.\n")

with open(ARCHIVO, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["hora", "segundo", "cpu_porcentaje",
                    "ram_usada_mb", "ram_porcentaje"])

    segundo = 0
    try:
        while True:
            cpu = psutil.cpu_percent(interval=1)  # bloquea 1 seg y mide
            mem = psutil.virtual_memory()
            ram_mb = mem.used / (1024 * 1024)
            hora = datetime.now().strftime("%H:%M:%S")

            print(
                f"[{hora}] t={segundo}s | CPU: {cpu:5.1f}% | RAM: {ram_mb:7.0f} MB ({mem.percent}%)")
            writer.writerow([hora, segundo, cpu, round(ram_mb), mem.percent])
            f.flush()  # escribe al disco cada segundo, por si cortas con Ctrl+C
            segundo += 1
    except KeyboardInterrupt:
        print(f"\nMedicion detenida. Datos guardados en {ARCHIVO}")
