# Tesis BTC Windows CUDA

Repositorio de entrenamiento de modelos de aprendizaje profundo en **Windows con CUDA** para el proyecto de tesis de predicción del precio de Bitcoin.

## Descripción

Este repositorio contiene la parte del proyecto orientada al entrenamiento y evaluación de modelos de deep learning usando aceleración por GPU en Windows.

Forma parte de una arquitectura de tesis dividida en tres componentes principales:

1. **Pipeline local**
2. **Pipeline WSL/Docker**
3. **Entrenamiento en Windows con CUDA**

Este repositorio corresponde al componente **Windows CUDA**.

---

## Objetivo

Ejecutar experimentos de modelado predictivo sobre series temporales de Bitcoin mediante modelos de aprendizaje profundo, aprovechando el uso de GPU con CUDA para acelerar el entrenamiento.

---

## Tecnologías utilizadas

- **Windows**
- **Python**
- **PyTorch**
- **CUDA**
- **Git y GitHub**

---

## Estructura del proyecto

```text
tesis_btc_windows/
├── src/
├── requirements_cuda_windows.txt
├── install_torch_cuda118.ps1
├── .gitignore
└── README.md
Descripción de archivos principales
src/: scripts de generación de escenarios, entrenamiento, evaluación y segmentación de resultados.
requirements_cuda_windows.txt: dependencias del entorno Python para este repositorio.
install_torch_cuda118.ps1: script de apoyo para instalar PyTorch con soporte CUDA.
.gitignore: exclusión de entornos virtuales, modelos, salidas, logs y archivos locales no necesarios.
Scripts principales

Dentro de src/ se incluyen scripts como:

09_generar_escenarios_1h.py
10_common_training_pipeline.py
11_baseline_persistencia.py
12_entrenar_lstm.py
13_entrenar_gru.py
14_entrenar_cnn1d.py
15_entrenar_cnn_lstm.py
16_run_experimentos.py
17_segmentar_resultados_dashboard.py
Requisitos previos

Antes de usar este repositorio, se recomienda contar con:

Windows 10/11
Python instalado
GPU NVIDIA compatible
CUDA correctamente configurado
Entorno virtual de Python
Git configurado
Instalación del entorno
1. Crear entorno virtual
python -m venv .venv_pt20
2. Activar entorno virtual
.\.venv_pt20\Scripts\activate
3. Instalar dependencias
pip install -r requirements_cuda_windows.txt
4. Instalar PyTorch con CUDA

Si aplica a tu entorno, se puede usar:

.\install_torch_cuda118.ps1
Ejecución

La ejecución depende del experimento o script que se quiera correr.

Ejemplo general
python .\src\11_baseline_persistencia.py
python .\src\12_entrenar_lstm.py
python .\src\13_entrenar_gru.py
python .\src\14_entrenar_cnn1d.py
python .\src\15_entrenar_cnn_lstm.py
python .\src\16_run_experimentos.py
Alcance del repositorio

Este repositorio puede incluir, entre otros:

generación de escenarios de evaluación
baseline de persistencia
entrenamiento de modelos LSTM
entrenamiento de modelos GRU
entrenamiento de modelos CNN 1D
entrenamiento de modelos híbridos CNN-LSTM
ejecución comparativa de experimentos
segmentación de resultados para visualización y dashboards
Archivos excluidos del repositorio

Por control de tamaño y orden del proyecto, no se incluyen en GitHub:

entornos virtuales
modelos entrenados
outputs generados
logs
configuraciones locales de VS Code
datos pesados

Esto se controla mediante el archivo .gitignore.

Relación con la tesis

Este repositorio forma parte del proyecto de tesis orientado a la comparación del rendimiento predictivo de modelos de aprendizaje profundo aplicados a la predicción del precio de Bitcoin.

El componente de preparación local de datos y el componente WSL/Docker se mantienen en repositorios separados.

Autor

Juan Andres Logacho Torres

Maestría en Ciencia de Datos y Big Data

Notas

Este repositorio documenta únicamente el componente de entrenamiento en Windows con CUDA del proyecto general.
Para la solución completa, revisar también los repositorios complementarios del pipeline local y del pipeline WSL/Docker.
