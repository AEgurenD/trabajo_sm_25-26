## Lista de comprobaciones

- [x] Comprobar funcionamiento de método LoadAudio()

## Pruebas con los filtros

* demo_lowpass.wav → Paso bajo, Fc1 = 1000 Hz, orden 5
Antes suenas un grave limpio más dos chirridos agudos muy molestos (8 kHz y 12 kHz). Al aplicar el filtro los chirridos desaparecen completamente y solo queda el tono grave.

* demo_highpass.wav → Paso alto, Fc1 = 500 Hz, orden 5
Antes domina un zumbido eléctrico muy grave y pesado (60 Hz + armónicos) que casi tapa un tono de 1500 Hz que hay debajo. Al filtrar, el zumbido desaparece y el tono medio queda limpio.

* demo_bandpass.wav → Paso banda, Fc1 = 800 Hz, Fc2 = 2000 Hz, orden 5
Hay 5 tonos simultáneos repartidos por todo el espectro (100, 500, 1500, 4000, 10000 Hz), todos a igual volumen. Suena como un acorde disonante y denso. Al filtrar solo sobrevive el tono de 1500 Hz, que estaba exactamente dentro de la banda. El resultado es un pitido único y limpio.

* demo_bandstop.wav → Banda eliminada, Fc1 = 45 Hz, Fc2 = 110 Hz, orden 5
Simula una grabación contaminada por interferencia de la red eléctrica europea (50 Hz + su armónico en 100 Hz). El zumbido es muy dominante y tapa casi todo. Al aplicar el notch desaparece el zumbido y quedan los tres tonos de voz (200, 800, 2000 Hz) perfectamente limpios.