# 🍷 Wine Quality ML Dashboard App

Una aplicación interactiva construida con **Streamlit**, alojada en **Streamlit Cloud**, que explora las capacidades de **Machine Learning** (regresión, clasificación y clustering) usando el famoso dataset de *Wine Quality*. El foco está en mostrar las fortalezas de las principales librerías de visualización en Python: **Matplotlib**, **Seaborn**, **Plotly** y **Altair**, con ejemplos prácticos y dashboards comparativos.

## 📁 Estructura del Proyecto

```
visual_py/
├── .visual/                  # Entorno virtual local
├── data/                     # Datos utilizados en la aplicación
│   └── winequality.csv       # Dataset original
├── notebooks/                # Notebooks para exploración de datos
│   ├── regression.ipynb      # Exploración de regresión
│   ├── classification.ipynb  # Exploración de clasificación
│   └── clustering.ipynb      # Exploración de clustering
├── dashboards/               # Dashboards para visualización
│   ├── matplotlib.py         # Dashboard con Matplotlib
│   ├── seaborn.py            # Dashboard con Seaborn
│   ├── plotly.py             # Dashboard con Plotly
│   └── altair.py             # Dashboard con Altair
├── utils/                    # Funciones utilitarias
│   └── json_export.py        # Exporta datos a JSON para Streamlit Cloud
├── app.py                    # App principal en Streamlit
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Documentación del proyecto
```
## 🚀 Objetivos del Proyecto

- Aplicar técnicas de ML (regresión, clasificación, clustering) sobre el dataset *Wine Quality*.
- Visualizar los resultados con las principales librerías gráficas de Python.
- Comparar sus fortalezas, estilos y casos de uso ideales.
- Generar dashboards interactivos y exportables.
- Documentar cada paso con explicaciones claras y profundas.

## 📊 Librerías de Visualización

| Librería   | Fortalezas | Casos ideales |
|------------|------------|----------------|
| Matplotlib | Control total, gráficos básicos | Publicaciones científicas, gráficos estáticos |
| Seaborn    | Estética mejorada, integración con pandas | Análisis exploratorio rápido |
| Plotly     | Interactividad, zoom, hover | Dashboards web, presentaciones |
| Altair     | Declarativa, basada en Vega | Visualizaciones estadísticas complejas |

Cada dashboard incluye:
- Gráficos de dispersión, histogramas, boxplots, heatmaps, etc.
- Comparación de resultados de modelos ML.
- Exportación de datos en formato JSON para Streamlit Cloud.

## 🧪 Requisitos

- Python 3.10+
- Git
- Streamlit
- Entorno virtual `.visual`

## 🛠️ Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/wine-ml-dashboard.git
cd wine-ml-dashboard

# Crear entorno virtual
python -m venv .visual
source .visual/bin/activate  # En Windows: .visual\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la app
streamlit run app.py
```

☁️ Despliegue en Streamlit Cloud
Una vez que tengas tu cuenta en Streamlit Cloud, sube el repositorio completo. Asegúrate de incluir:

- app.py
- requirements.txt
- data/winequality.csv
- utils/json_export.py
🔗 Enlace a la app en Streamlit Cloud:

Streamlit Cloud Logo
Accede a la app aquí ← Reemplaza con tu URL cuando esté disponible

📦 Exportación de Datos
La app genera una estructura JSON con los datos procesados y visualizados, lista para ser usada en la nube:

```json
{
  "model": "RandomForestClassifier",
  "metrics": {
    "accuracy": 0.87,
    "f1_score": 0.85
  },
  "visualizations": {
    "plotly": {...},
    "seaborn": {...},
    "altair": {...},
    "matplotlib": {...}
  }
}
```

🤝 Contribuciones
Este proyecto busca inspirar a otros desarrolladores autodidactas. Si tienes ideas, mejoras o quieres sumar tus visualizaciones, ¡bienvenido!

```bash
# Forkea el repo
# Crea tu rama
git checkout -b mejora-visualizacion
# Haz tus cambios y súbelos
```


🧠 Autor
Daniel Mardones
Técnico industrial de campo, integrador IoT y desarrollador autodidacta.
Apasionado por el aprendizaje práctico, la visualización de datos y el desarrollo colaborativo.
📍 Temuco, Chile

---

## 📄 Licencia y Uso

Este repositorio es **público** pero **no cuenta con ninguna licencia**. Si deseas utilizar el código, por favor contáctame personalmente a través de los siguientes enlaces:

<a href="https://www.linkedin.com/in/daniel-andres-mardones-sanhueza-27b73777" target="_blank" style="text-decoration:none;">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="24" style="vertical-align:middle;" /> LinkedIn
</a>

<a href="https://github.com/Denniels" target="_blank" style="text-decoration:none;">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="24" style="vertical-align:middle;" /> GitHub
</a>

---

