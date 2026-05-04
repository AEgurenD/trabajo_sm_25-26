# Procesado de Audio — Sistemas Multimedia

Aplicación de escritorio en Python/PySide6 para carga, visualización, filtrado y compresión de audio WAV y formatos comprimidos.

## Características

- **Carga** de WAV, MP3, OGG, FLAC, AIFF, AAC
- **Reproducción** con barra de progreso, seek y control de volumen
- **Conversión** Mono ↔ Estéreo
- **Visualización** de onda temporal, espectro FFT y espectrograma (matplotlib)
- **Filtros** Butterworth: paso bajo, paso alto, paso banda, banda eliminada
- **Análisis de calidad** automático tras cada operación: SNR, THD, rango dinámico
- **Compresión clásica**: OGG Vorbis (con pérdida), FLAC (sin pérdida) — vía `soundfile`, sin dependencias externas
- **Compresión neuronal**: EnCodec (Meta) a 1.5 – 24 kbps — requiere `torch` y `encodec`

## Requisitos del sistema

- Python **3.12** recomendado (3.13+ rompe algunas dependencias opcionales)
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
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Estructura del proyecto

```
.
├── app.py                   # Punto de entrada
├── mainwindow.ui            # Diseño de la interfaz Qt
├── generate_demo_audio.py   # Genera archivos de prueba para los filtros
├── audio/
│   ├── _input/              # Coloca aquí los archivos de entrada
│   └── _output/             # Los archivos procesados se guardan aquí
├── environment.yml
├── requirements.txt
└── README.md
```

## Uso

```bash
python app.py
```

Para generar los audios de prueba de los filtros:

```bash
python generate_demo_audio.py
```

Los archivos se crean en `audio/_input/`.

## Archivos de demo para filtros

| Archivo | Filtro recomendado | Ajuste sugerido | Efecto audible |
|---|---|---|---|
| `demo_lowpass.wav` | Paso bajo | Fc1 = 1000 Hz, Orden = 5 | Desaparecen los chirridos agudos |
| `demo_highpass.wav` | Paso alto | Fc1 = 500 Hz, Orden = 5 | Desaparece el zumbido grave |
| `demo_bandpass.wav` | Paso banda | Fc1 = 800 Hz, Fc2 = 2000 Hz, Orden = 5 | Solo sobrevive el tono de 1500 Hz |
| `demo_bandstop.wav` | Banda eliminada | Fc1 = 45 Hz, Fc2 = 110 Hz, Orden = 5 | Desaparece el zumbido eléctrico de 50 Hz |

## Compresión

### Clásica (OGG y FLAC)
No requiere ninguna instalación adicional. `soundfile` gestiona ambos formatos de forma nativa.

- **OGG Vorbis**: compresión con pérdida, ratio típico 5–10x
- **FLAC**: compresión sin pérdida, ratio típico 2x, audio idéntico al original

### Neuronal (EnCodec)
Requiere `torch` y `encodec`. La primera ejecución descarga los pesos del modelo (~50 MB).

```bash
pip install torch encodec
```

Bandwidths disponibles: 1.5 / 3.0 / 6.0 / 12.0 / 24.0 kbps.
A mayor bandwidth, mayor calidad y menor ratio de compresión.

## Notas

- La exportación a **MP3 no está disponible** en esta versión ya que requiere `ffmpeg` instalado en el sistema como dependencia externa.
- Para GPU con EnCodec, sustituir `cpuonly` por `pytorch-cuda=12.1` en `environment.yml` o usar el índice CUDA en pip: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
