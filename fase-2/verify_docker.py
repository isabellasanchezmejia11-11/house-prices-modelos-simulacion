"""Verifica la imagen y la persistencia de archivos fuera del contenedor."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import tempfile


def main():
    if shutil.which('docker') is None:
        raise SystemExit('Docker no está instalado o no se encuentra en PATH.')
    root = Path(__file__).resolve().parent
    subprocess.run(['docker', 'build', '-t', 'house-prices-fase2:entrega-2', str(root)], check=True)
    with tempfile.TemporaryDirectory(prefix='house-prices-docker-') as directory:
        workspace = Path(directory)
        shutil.copy(root.parent / 'fase-1/train.csv', workspace / 'train.csv')
        shutil.copy(root / 'data/example_prediction.csv', workspace / 'input.csv')
        base = ['docker', 'run', '--rm', '-v', f'{workspace}:/data',
                'house-prices-fase2:entrega-2']
        for version, seed in [('docker-v1', '42'), ('docker-v2', '43')]:
            subprocess.run(base + ['train.py', '--data', '/data/train.csv', '--target', 'SalePrice',
                                  '--output-dir', '/data/models', '--version', version,
                                  '--seed', seed], check=True)
            for name in ['model.joblib', 'metrics.json', 'metadata.json']:
                if not (workspace / 'models' / version / name).is_file():
                    raise RuntimeError(f'No persistió {version}/{name}.')
            json.loads((workspace / 'models' / version / 'metrics.json').read_text())
            subprocess.run(base + ['predict.py', '--model', f'/data/models/{version}',
                                  '--input', '/data/input.csv', '--output', f'/data/{version}.csv'],
                           check=True)
            with (workspace / f'{version}.csv').open() as source:
                rows = list(csv.DictReader(source))
            if len(rows) != 1 or not rows[0].get('SalePricePrediction'):
                raise RuntimeError('La salida no contiene la predicción esperada.')
        invalid = subprocess.run(base + ['predict.py', '--model', '/data/missing',
                                         '--input', '/data/input.csv', '--output', '/data/invalid.csv'],
                                 capture_output=True, text=True)
        if invalid.returncode == 0 or (workspace / 'invalid.csv').exists():
            raise RuntimeError('La entrada inválida no se rechazó correctamente.')
    print('Docker: construcción, dos versiones, predicción y persistencia verificadas.')


if __name__ == '__main__':
    main()
