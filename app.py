from PySide6.QtCore import Qt, QFile, QUrl
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq
import soundfile as sf
import matplotlib
matplotlib.use("Agg")                  # backend sin ventana
import matplotlib.pyplot as plt
import matplotlib.figure as mplfig
import cv2
import numpy as np
import tempfile
import sys
import os

PLOT_W = 820   # píxeles de la figura
PLOT_H = 520


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ── Cargar UI ──
        ui_path = os.path.join(os.path.dirname(__file__), "mainwindow.ui")
        ui_file = QFile(ui_path)
        ui_file.open(QFile.ReadOnly)
        self.ui = QUiLoader().load(ui_file, self)
        ui_file.close()

        # ── Directorios ──
        cwd = os.getcwd()
        self.directorio_entrada = os.path.join(cwd, 'audio', '_input')
        self.directorio_salida  = os.path.join(cwd, 'audio', '_output')

        # ── Estado del audio ──
        self.audio_data  = None   # np.ndarray float32 [-1, 1]
        self.sample_rate = None
        self.audio_path  = None
        self.n_channels  = None

        # ── Reproducción ──
        self._player       = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(1.0)
        self._temp_wav      = None
        self._seeking       = False   # evita feedback slider ↔ player

        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)

        # ── Compresión neuronal (EnCodec) ──
        self._neural_codes = None
        self._neural_bw    = 6.0

        # ── Conexiones ──
        self.ui.btnCargar.clicked.connect(self._on_cargar)
        self.ui.btnGuardar.clicked.connect(self._on_guardar)
        self.ui.btnPlay.clicked.connect(self._on_play)
        self.ui.btnStop.clicked.connect(self._on_stop)
        self.ui.sliderProgreso.sliderPressed.connect(self._on_slider_pressed)
        self.ui.sliderProgreso.sliderReleased.connect(self._on_slider_released)
        self.ui.sliderVolumen.valueChanged.connect(self._on_volume_changed)

        self.ui.btnStereo2Mono.clicked.connect(self._on_stereo2mono)
        self.ui.btnMono2Stereo.clicked.connect(self._on_mono2stereo)

        self.ui.btnVerOnda.clicked.connect(self._on_ver_onda)
        self.ui.btnVerEspectro.clicked.connect(self._on_ver_espectro)
        self.ui.btnVerEspectrograma.clicked.connect(self._on_ver_espectrograma)

        self.ui.btnLowPass.clicked.connect(self._on_lowpass)
        self.ui.btnHighPass.clicked.connect(self._on_highpass)
        self.ui.btnBandPass.clicked.connect(self._on_bandpass)
        self.ui.btnBandStop.clicked.connect(self._on_bandstop)

        self.ui.btnCompOGG.clicked.connect(self._on_comp_ogg)
        self.ui.btnCompFLAC.clicked.connect(self._on_comp_flac)

        self.ui.btnEncode.clicked.connect(self._on_encode_neural)
        self.ui.btnDecode.clicked.connect(self._on_decode_neural)

        self.ui.show()
        self.resize(self.ui.geometry().width(), self.ui.geometry().height())

    # =================================================================
    # HELPERS DE UI
    # =================================================================

    def _set_status(self, texto):
        self.ui.lblStatus.setText(texto)

    @staticmethod
    def _fmt_ms(ms):
        """Convierte milisegundos a mm:ss."""
        s = ms // 1000
        return f"{s // 60:02d}:{s % 60:02d}"

    def _refresh_info_labels(self):
        if self.audio_data is None:
            return
        dur = self.audio_data.shape[0] / self.sample_rate
        h, m, s = int(dur // 3600), int((dur % 3600) // 60), dur % 60
        self.ui.lblNombreVal.setText(os.path.basename(self.audio_path))
        self.ui.lblSRVal.setText(f"{self.sample_rate} Hz")
        self.ui.lblCanVal.setText("Mono" if self.n_channels == 1 else "Estéreo")
        self.ui.lblDurVal.setText(f"{h:02d}:{m:02d}:{s:06.3f}")

    def _refresh_calidad(self):
        """Recalcula y muestra SNR, THD y DR automáticamente."""
        self.ui.lblSNRVal.setText(f"SNR: {self.compute_snr()['etiqueta']}")
        self.ui.lblTHDVal.setText(f"THD: {self.compute_thd()['etiqueta']}")
        self.ui.lblDRVal.setText(f"DR:  {self.compute_dynamic_range()['etiqueta']}")

    def _mostrar_figura(self, fig, titulo=""):
        """
        Renderiza una matplotlib Figure a imagen RGB con cv2
        y la muestra en el QLabel central.
        """
        if fig is None:
            self._set_status("No hay audio cargado para visualizar.")
            return
        self.ui.lblGraficaTitulo.setText(titulo)

        # 1. Dibujar la figura en un canvas en memoria
        fig.set_size_inches(PLOT_W / 100, PLOT_H / 100)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        plt.close(fig)

        # 2. Mostrar en QLabel
        h, w, ch = img.shape
        qimg   = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.ui.lblGrafica.width(), self.ui.lblGrafica.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.lblGrafica.setPixmap(pixmap)

    def _preparar_temp_wav(self):
        """Escribe el audio actual a un WAV temporal para QMediaPlayer."""
        if self.audio_data is None:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, self.audio_data, self.sample_rate)
        self._temp_wav = tmp.name
        self._player.setSource(QUrl.fromLocalFile(self._temp_wav))

    def _mostrar_resultado_compresion(self, res):
        if res["ok"]:
            self.ui.lblCompOrig.setText(f"Original:   {res['tamanio_original_kb']} KB")
            self.ui.lblCompOut.setText(f"Comprimido: {res['tamanio_comprimido_kb']} KB")
            self.ui.lblCompRatio.setText(f"Ratio: {res['ratio']}x")
        self._set_status(res["mensaje"])

    # =================================================================
    # SLOTS — Reproducción
    # =================================================================

    def _on_play(self):
        if self._temp_wav is None:
            self._set_status("Carga un archivo primero.")
            self.ui.btnPlay.setChecked(False)
            return
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_stop(self):
        self._player.stop()

    def _on_playback_state_changed(self, state):
        playing = (state == QMediaPlayer.PlayingState)
        self.ui.btnPlay.setText("⏸  Pausa" if playing else "▶  Play")
        self.ui.btnPlay.setChecked(playing)

    def _on_position_changed(self, pos_ms):
        """Actualiza slider y etiqueta de tiempo sin generar bucle de feedback."""
        if self._seeking:
            return
        dur = self._player.duration()
        if dur > 0:
            self.ui.sliderProgreso.setValue(int(pos_ms / dur * 1000))
        self.ui.lblTiempoActual.setText(self._fmt_ms(pos_ms))

    def _on_duration_changed(self, dur_ms):
        self.ui.lblTiempoTotal.setText(self._fmt_ms(dur_ms))

    def _on_slider_pressed(self):
        self._seeking = True

    def _on_slider_released(self):
        dur = self._player.duration()
        if dur > 0:
            pos = int(self.ui.sliderProgreso.value() / 1000 * dur)
            was_playing = (self._player.playbackState() == QMediaPlayer.PlayingState)
            self._player.setPosition(pos)
            # En algunos backends setPosition detiene la reproducción; la reanudamos
            if was_playing:
                self._player.play()
        self._seeking = False

    def _on_volume_changed(self, value):
        self._audio_output.setVolume(value / 100.0)

    # =================================================================
    # SLOTS — Archivo
    # =================================================================

    def _on_cargar(self):
        if not self.load_audio():
            self._set_status("Carga cancelada o fallida.")
            return
        self._refresh_info_labels()
        self._refresh_calidad()
        self._preparar_temp_wav()
        self._set_status(f"Cargado: {os.path.basename(self.audio_path)}")

    def _on_guardar(self):
        res = self.save_audio()
        self._set_status(res["mensaje"])

    # =================================================================
    # SLOTS — Canales
    # =================================================================

    def _on_stereo2mono(self):
        res = self.stereo_2_mono()
        if res["ok"]:
            self.ui.lblCanVal.setText("Mono")
            self._refresh_calidad()
            self._preparar_temp_wav()
        self._set_status(res["mensaje"])

    def _on_mono2stereo(self):
        res = self.mono_2_stereo()
        if res["ok"]:
            self.ui.lblCanVal.setText("Estéreo")
            self._refresh_calidad()
            self._preparar_temp_wav()
        self._set_status(res["mensaje"])

    # =================================================================
    # SLOTS — Visualización
    # =================================================================

    def _on_ver_onda(self):
        self._mostrar_figura(self.plot_waveform(), "Onda temporal")

    def _on_ver_espectro(self):
        self._mostrar_figura(self.plot_spectrum(), "Espectro de frecuencia")

    def _on_ver_espectrograma(self):
        self._mostrar_figura(self.plot_spectrogram(), "Espectrograma")

    # =================================================================
    # SLOTS — Filtros
    # =================================================================

    def _aplicar_filtro(self, res):
        if res["ok"]:
            self._refresh_calidad()
            self._preparar_temp_wav()
        self._set_status(res["mensaje"])

    def _on_lowpass(self):
        self._aplicar_filtro(self.apply_lowpass(
            self.ui.spinFc1.value(), self.ui.spinOrden.value()))

    def _on_highpass(self):
        self._aplicar_filtro(self.apply_highpass(
            self.ui.spinFc1.value(), self.ui.spinOrden.value()))

    def _on_bandpass(self):
        self._aplicar_filtro(self.apply_bandpass(
            self.ui.spinFc1.value(), self.ui.spinFc2.value(), self.ui.spinOrden.value()))

    def _on_bandstop(self):
        self._aplicar_filtro(self.apply_bandstop(
            self.ui.spinFc1.value(), self.ui.spinFc2.value(), self.ui.spinOrden.value()))

    # =================================================================
    # SLOTS — Compresión
    # =================================================================

    def _on_comp_ogg(self):
        self._mostrar_resultado_compresion(self.compress_ogg())

    def _on_comp_flac(self):
        self._mostrar_resultado_compresion(self.compress_flac())

    def _on_encode_neural(self):
        res = self.encode_neural(float(self.ui.spinBW.currentText()))
        if res["ok"]:
            self.ui.lblNeuralRatio.setText(f"Ratio: {res['ratio']}x")
        self._set_status(res["mensaje"])

    def _on_decode_neural(self):
        res = self.decode_neural()
        if res["ok"]:
            self._refresh_info_labels()
            self._refresh_calidad()
            self._preparar_temp_wav()
        self._set_status(res["mensaje"])

    # =================================================================
    # SOPORTE INTERNO
    # =================================================================

    def normalize_audio(self, data):
        if data.dtype == np.int16:
            return data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            return data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            return (data.astype(np.float32) - 128.0) / 128.0
        return data.astype(np.float32)

    def get_active_audio(self):
        """Devuelve array 1-D float32 para análisis/visualización."""
        if self.audio_data is None:
            return None
        return self.audio_data if self.audio_data.ndim == 1 \
            else self.audio_data.mean(axis=1).astype(np.float32)

    # =================================================================
    # CARGA Y GUARDADO
    # =================================================================

    def load_audio(self):
        """Devuelve True si se cargó correctamente."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir archivo de audio", self.directorio_entrada,
            "Audio Files (*.wav *.mp3 *.ogg *.flac *.aiff *.aac)")
        if not file_path:
            return False
        try:
            ext = os.path.splitext(file_path)[1].lower()
            sr, data = wavfile.read(file_path) if ext == ".wav" \
                else sf.read(file_path, always_2d=False)[::-1]
            self.audio_data  = self.normalize_audio(data)
            self.sample_rate = sr
            self.audio_path  = file_path
            self.n_channels  = 1 if self.audio_data.ndim == 1 else self.audio_data.shape[1]
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar audio", str(e))
            return False

    def save_audio(self):
        if self.audio_data is None:
            return {"ok": False, "mensaje": "No hay audio cargado."}
        os.makedirs(self.directorio_salida, exist_ok=True)
        base  = os.path.splitext(os.path.basename(self.audio_path))[0]
        ruta  = os.path.join(self.directorio_salida, base + "_out.wav")
        try:
            sf.write(ruta, self.audio_data, self.sample_rate)
            return {"ok": True, "mensaje": f"Guardado en {ruta}"}
        except Exception as e:
            return {"ok": False, "mensaje": str(e)}

    # =================================================================
    # CONVERSIÓN DE CANALES
    # =================================================================

    def stereo_2_mono(self):
        if self.audio_data is None:
            return {"ok": False, "mensaje": "No hay audio cargado.", "n_channels": None}
        if self.audio_data.ndim == 1:
            return {"ok": False, "mensaje": "El audio ya es mono.", "n_channels": 1}
        self.audio_data = self.audio_data.mean(axis=1).astype(np.float32)
        self.n_channels = 1
        return {"ok": True, "mensaje": "Convertido a mono.", "n_channels": 1}

    def mono_2_stereo(self):
        if self.audio_data is None:
            return {"ok": False, "mensaje": "No hay audio cargado.", "n_channels": None}
        if self.audio_data.ndim == 2:
            return {"ok": False, "mensaje": "El audio ya es estéreo.", "n_channels": 2}
        self.audio_data = np.stack([self.audio_data, self.audio_data], axis=1)
        self.n_channels = 2
        return {"ok": True, "mensaje": "Convertido a estéreo.", "n_channels": 2}

    # =================================================================
    # VISUALIZACIÓN — matplotlib (sin kaleido)
    # =================================================================

    def _fig(self):
        """Devuelve una nueva figura oscura lista para dibujar."""
        fig, ax = plt.subplots(facecolor="#1e1e2e")
        ax.set_facecolor("#1e1e2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        return fig, ax

    def plot_waveform(self):
        signal = self.get_active_audio()
        if signal is None:
            return None
        t = np.linspace(0, len(signal) / self.sample_rate, len(signal))
        # Decimar para que la figura no sea lenta con archivos largos
        step = max(1, len(signal) // 10000)
        fig, ax = self._fig()
        ax.plot(t[::step], signal[::step], color="#4e8ef7", linewidth=0.8)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud")
        ax.set_title("Onda temporal")
        fig.tight_layout()
        return fig

    def plot_spectrum(self):
        signal = self.get_active_audio()
        if signal is None:
            return None
        N     = len(signal)
        yf    = np.abs(rfft(signal)) * 2 / N
        xf    = rfftfreq(N, 1 / self.sample_rate)
        yf_db = 20 * np.log10(yf + 1e-10)
        fig, ax = self._fig()
        ax.plot(xf, yf_db, color="#e05c5c", linewidth=0.8)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud (dB)")
        ax.set_title("Espectro de frecuencia")
        fig.tight_layout()
        return fig

    def plot_spectrogram(self, n_fft=1024, hop=512):
        signal = self.get_active_audio()
        if signal is None:
            return None
        window = np.hanning(n_fft)
        frames = [np.abs(rfft(signal[i:i + n_fft] * window))
                  for i in range(0, len(signal) - n_fft, hop)]
        S_db  = 20 * np.log10(np.array(frames).T + 1e-10)
        freqs = rfftfreq(n_fft, 1 / self.sample_rate)
        times = np.arange(S_db.shape[1]) * hop / self.sample_rate
        fig, ax = self._fig()
        img = ax.pcolormesh(times, freqs, S_db, cmap="magma", shading="auto")
        fig.colorbar(img, ax=ax, label="dB")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")
        ax.set_title("Espectrograma")
        fig.tight_layout()
        return fig

    # =================================================================
    # FILTROS
    # =================================================================

    def _butter_filter(self, data, cutoff, btype, order):
        nyq = self.sample_rate / 2.0
        wn  = [c / nyq for c in cutoff] if isinstance(cutoff, (list, tuple)) else cutoff / nyq
        sos = butter(order, wn, btype=btype, output='sos')
        if data.ndim == 2:
            return np.stack([sosfilt(sos, data[:, ch])
                             for ch in range(data.shape[1])], axis=1).astype(np.float32)
        return sosfilt(sos, data).astype(np.float32)

    def apply_lowpass(self, cutoff_hz=4000, order=5):
        if self.audio_data is None:
            return {"ok": False, "mensaje": "No hay audio cargado."}
        self.audio_data = self._butter_filter(self.audio_data, cutoff_hz, 'low', order)
        return {"ok": True, "mensaje": f"Paso bajo: {cutoff_hz} Hz, orden {order}."}

    def apply_highpass(self, cutoff_hz=300, order=5):
        if self.audio_data is None:
            return {"ok": False, "mensaje": "No hay audio cargado."}
        self.audio_data = self._butter_filter(self.audio_data, cutoff_hz, 'high', order)
        return {"ok": True, "mensaje": f"Paso alto: {cutoff_hz} Hz, orden {order}."}

    def apply_bandpass(self, low_hz=300, high_hz=4000, order=5):
        if self.audio_data is None:
            return {"ok": False, "mensaje": "No hay audio cargado."}
        self.audio_data = self._butter_filter(self.audio_data, [low_hz, high_hz], 'bandpass', order)
        return {"ok": True, "mensaje": f"Paso banda: {low_hz}–{high_hz} Hz, orden {order}."}

    def apply_bandstop(self, low_hz=50, high_hz=60, order=5):
        if self.audio_data is None:
            return {"ok": False, "mensaje": "No hay audio cargado."}
        self.audio_data = self._butter_filter(self.audio_data, [low_hz, high_hz], 'bandstop', order)
        return {"ok": True, "mensaje": f"Banda eliminada: {low_hz}–{high_hz} Hz, orden {order}."}

    # =================================================================
    # ANÁLISIS DE CALIDAD  (automáticos)
    # =================================================================

    def _signal_for_quality(self):
        """
        Devuelve la señal sobre la que calcular las métricas.
        - Mono  → el array tal cual (1-D)
        - Estéreo → canal L (columna 0), NO la media, para que el
          cambio estéreo↔mono sea perceptible en las métricas.
        """
        if self.audio_data is None:
            return None
        if self.audio_data.ndim == 1:
            return self.audio_data
        return self.audio_data[:, 0].astype(np.float32)  # canal L

    def compute_snr(self):
        signal = self._signal_for_quality()
        if signal is None:
            return {"snr_db": None, "etiqueta": "Sin audio"}

        N           = len(signal)
        yf          = np.abs(rfft(signal)) ** 2 * 2 / N
        sig_pow     = float(np.mean(signal ** 2))
        noise_floor = float(np.median(yf))
        noise_pow   = noise_floor * (N // 2) / N

        if noise_pow <= 0 or sig_pow <= 0:
            return {"snr_db": 0.0, "etiqueta": "0.00 dB"}

        snr = 10 * np.log10(sig_pow / noise_pow)
        return {"snr_db": round(float(snr), 2), "etiqueta": f"{snr:.2f} dB"}

    def compute_thd(self, n_harmonics=5):
        signal = self._signal_for_quality()
        if signal is None:
            return {"thd_pct": None, "fundamental_hz": None, "etiqueta": "Sin audio"}

        N  = len(signal)
        yf = np.abs(rfft(signal)) * 2 / N
        xf = rfftfreq(N, 1 / self.sample_rate)
        f0 = float(xf[np.argmax(yf)])

        def amp(f):
            return float(yf[np.argmin(np.abs(xf - f))])

        thd = (np.sqrt(sum(amp(f0 * k) ** 2 for k in range(2, n_harmonics + 1)))
               / (amp(f0) + 1e-10)) * 100
        return {"thd_pct": round(float(thd), 4), "fundamental_hz": round(f0, 2),
                "etiqueta": f"{thd:.4f} % (f0={f0:.1f} Hz)"}

    def compute_dynamic_range(self):
        signal = self._signal_for_quality()
        if signal is None:
            return {"dr_db": None, "etiqueta": "Sin audio"}

        pico     = np.max(np.abs(signal))
        non_zero = np.abs(signal[np.abs(signal) > 1e-10])
        minimo   = np.min(non_zero) if len(non_zero) > 0 else 1e-10
        pico_db  = 20 * np.log10(pico   + 1e-10)
        min_db   = 20 * np.log10(minimo + 1e-10)
        dr       = pico_db - min_db
        return {"dr_db": round(float(dr), 2),
                "etiqueta": f"{dr:.2f} dB  (pico {pico_db:.1f} / mín {min_db:.1f} dB)"}

    # =================================================================
    # COMPRESIÓN CLÁSICA
    # =================================================================

    def _comp_result(self, ruta_out, fmt):
        sz_orig = self.audio_data.size * self.audio_data.itemsize / 1024
        sz_comp = os.path.getsize(ruta_out) / 1024
        return {"ok": True, "ruta": ruta_out,
                "tamanio_original_kb"  : round(sz_orig, 2),
                "tamanio_comprimido_kb": round(sz_comp, 2),
                "ratio"  : round(sz_orig / (sz_comp + 1e-6), 2),
                "mensaje": f"{fmt} guardado en {ruta_out}"}

    def _comp_error(self, e):
        return {"ok": False, "ruta": "", "mensaje": str(e),
                "tamanio_original_kb": None, "tamanio_comprimido_kb": None, "ratio": None}



    def compress_ogg(self):
        if self.audio_data is None:
            return self._comp_error(Exception("No hay audio cargado."))
        try:
            os.makedirs(self.directorio_salida, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.audio_path))[0]
            ruta = os.path.join(self.directorio_salida, base + "_compressed.ogg")
            sf.write(ruta, self.audio_data, self.sample_rate, format='OGG', subtype='VORBIS')
            return self._comp_result(ruta, "OGG")
        except Exception as e:
            return self._comp_error(e)

    def compress_flac(self):
        if self.audio_data is None:
            return self._comp_error(Exception("No hay audio cargado."))
        try:
            os.makedirs(self.directorio_salida, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.audio_path))[0]
            ruta = os.path.join(self.directorio_salida, base + "_compressed.flac")
            sf.write(ruta, self.audio_data, self.sample_rate, format='FLAC')
            return self._comp_result(ruta, "FLAC")
        except Exception as e:
            return self._comp_error(e)

    # =================================================================
    # COMPRESIÓN NEURONAL  (EnCodec)
    # =================================================================

    def encode_neural(self, bandwidth=6.0):
        """
        Codifica el audio con EnCodec 24kHz.
        bandwidth: 1.5 | 3.0 | 6.0 | 12.0 | 24.0  kbps
        """
        signal = self.get_active_audio()
        if signal is None:
            return {"ok": False, "ratio": None, "mensaje": "No hay audio cargado."}
        try:
            import torch
            from encodec import EncodecModel
            from encodec.utils import convert_audio

            os.makedirs(self.directorio_salida, exist_ok=True)
            base   = os.path.splitext(os.path.basename(self.audio_path))[0]
            tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(0)  # (1, 1, N)

            ec = EncodecModel.encodec_model_24khz()
            ec.set_target_bandwidth(bandwidth)
            ec.eval()

            wav = convert_audio(tensor, self.sample_rate, ec.sample_rate, ec.channels)
            with torch.no_grad():
                codes = torch.cat([c for c, _ in ec.encode(wav)], dim=-1)

            ruta = os.path.join(self.directorio_salida, f"{base}_encodec_codes.pt")
            torch.save(codes, ruta)

            sz_orig = len(signal) * signal.itemsize / 1024
            sz_cod  = os.path.getsize(ruta) / 1024
            self._neural_codes = codes
            self._neural_bw    = bandwidth
            return {"ok": True,
                    "ratio"  : round(sz_orig / (sz_cod + 1e-6), 2),
                    "mensaje": f"EnCodec {bandwidth} kbps — códigos guardados en {ruta}"}
        except Exception as e:
            return {"ok": False, "ratio": None, "mensaje": str(e)}

    def decode_neural(self):
        """Decodifica los últimos códigos EnCodec generados."""
        if self._neural_codes is None:
            return {"ok": False, "mensaje": "Ejecuta 'Codificar' primero."}
        try:
            import torch
            from encodec import EncodecModel

            ec = EncodecModel.encodec_model_24khz()
            ec.set_target_bandwidth(self._neural_bw)
            ec.eval()
            with torch.no_grad():
                audio_np = ec.decode([(self._neural_codes, None)]).squeeze().numpy()

            self.audio_data = audio_np.astype(np.float32)
            self.n_channels = 1 if audio_np.ndim == 1 else audio_np.shape[1]
            return {"ok": True, "mensaje": "Audio reconstruido con EnCodec."}
        except Exception as e:
            return {"ok": False, "mensaje": str(e)}


# =====================================================================
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app    = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('Proyecto de Audio')
    window.show()
    sys.exit(app.exec())