# Procesado de Audio — Sistemas Multimedia

Aplicación de escritorio en Python/PySide6 para carga, visualización, filtrado y compresión de audio WAV y formatos comprimidos.

## Características

- **Carga** de WAV, MP3, OGG, FLAC, AIFF, AAC
- **Reproducción** con barra de progreso y seek
- **Conversión** Mono ↔ Estéreo
- **Visualización** de onda temporal, espectro FFT y espectrograma (matplotlib)
- **Filtros** Butterworth: paso bajo, paso alto, paso banda, banda eliminada
- **Análisis de calidad** automático: SNR, THD, rango dinámico
- **Compresión clásica**: OGG (con pérdida), FLAC (sin pérdida)
- **Compresión neuronal**: EnCodec (Meta) a 1.5 – 24 kbps

## Requisitos

- Python **3.12** (DAC requiere ≤ 3.12; EnCodec funciona en 3.12)
- Conda (recomendado) o pip

## Instalación

### Opción A — Conda (recomendado)

```bash
conda env create -f environment.yml
conda activate audio_app
```

### Opción B — pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Estructura del proyecto

```
.
├── app.py                  # Punto de entrada
├── mainwindow.ui           # Diseño de la interfaz Qt
├── audio/
│   ├── _input/             # Coloca aquí los archivos de entrada
│   └── _output/            # Los archivos procesados se guardan aquí
├── environment.yml
├── requirements.txt
└── README.md
```

## Uso

```bash
python app.py
```

Los archivos de audio de prueba se pueden generar con:

```bash
python generate_demo_audio.py
```

## Filtros de demo

| Archivo | Filtro recomendado | Ajuste |
|---|---|---|
| `demo_lowpass.wav` | Paso bajo | Fc1 = 1000 Hz |
| `demo_highpass.wav` | Paso alto | Fc1 = 500 Hz |
| `demo_bandpass.wav` | Paso banda | Fc1 = 800 Hz, Fc2 = 2000 Hz |
| `demo_bandstop.wav` | Banda eliminada | Fc1 = 45 Hz, Fc2 = 110 Hz |

## Notas

- La compresión neuronal (EnCodec) requiere `torch`. La primera ejecución descarga los pesos del modelo (~50 MB).
- La exportación a MP3 requiere `ffmpeg` instalado en el sistema.
- Para GPU, sustituir `cpuonly` por `pytorch-cuda=12.1` en `environment.yml` o usar el índice CUDA en pip.
