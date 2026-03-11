# Stock Analysis (LSTM)

Proyecto para descargar históricos de acciones, entrenar un modelo LSTM y monitorear predicciones para múltiples tickers.

## Requisitos

```bash
pip install -r requirements.txt
```

## Archivos principales

- `get_stock_history.py`: descarga históricos desde Yahoo Finance.
- `train_lstm_model.py`: entrena modelos LSTM usando CSV históricos.
- `monitor_stocks.py`: abre ventanas por ticker con predicción de próximos días.
- `requirements.txt`: dependencias del proyecto.

## Flujo recomendado (orden de ejecución)

### 1) Descargar históricos

```bash
python get_stock_history.py AAPL MSFT OXY --period 10y --interval 1d --output-dir data/historical --stats
```

Ejemplo mínimo:

```bash
python get_stock_history.py OXY --output-dir data/historical
```

> Nota: usa `--output-dir data/historical` para que `train_lstm_model.py` encuentre los CSV automáticamente.

### 2) Entrenar modelos

```bash
python train_lstm_model.py
```

Salida esperada:

- Modelos `.h5` en `data/models/`
- Logs en `data/models/logs/`
- Imágenes de entrenamiento en `data/models/images/`

### 3) Monitorear múltiples tickers

```bash
python monitor_stocks.py --tickers OXY AAPL --days 7
```

Con modelo específico:

```bash
python monitor_stocks.py --tickers OXY AAPL IVVPESO.MX --days 7 --model data/models/TU_MODELO.h5
```

## Comandos útiles

Ver ayuda de cada script:

```bash
python get_stock_history.py --help
python monitor_stocks.py --help
```

## Estructura esperada

```text
Stock_Analysis/
├── data/
│   ├── historical/
│   └── models/
│       ├── images/
│       └── logs/
├── get_stock_history.py
├── train_lstm_model.py
├── monitor_stocks.py
└── requirements.txt
```

## Notas

- `monitor_stocks.py` usa multiprocessing: cada ticker abre su propia ventana.
- Si no pasas `--model`, el monitor usa el primer `.h5` disponible en `data/models/`.
- Esto es para fines educativos; no es asesoría financiera.
