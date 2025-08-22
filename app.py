import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt

# --- CSS personalizado para tarjetas y banner ---
st.markdown('''
	<style>
	.banner {
		background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
		color: #fff;
		padding: 2rem 1rem;
		border-radius: 12px;
		margin-bottom: 2rem;
		box-shadow: 0 4px 24px rgba(44,62,80,0.15);
		text-align: center;
	}
	.card {
		background: #f8f9fa;
		border-radius: 12px;
		box-shadow: 0 2px 12px rgba(44,62,80,0.08);
		padding: 1.5rem;
		margin-bottom: 2rem;
		border-left: 8px solid #2980b9;
	}
	.card-title {
		color: #2980b9;
		font-size: 1.3rem;
		font-weight: bold;
		margin-bottom: 0.5rem;
	}
	</style>
''', unsafe_allow_html=True)

# --- Banner principal ---
st.markdown('<div class="banner"><h1><span translate="no">🍷 Wine Quality ML Dashboard</span></h1><h3><span translate="no">Análisis Integral y Visualización avanzada</span></h3></div>', unsafe_allow_html=True)

# --- Cargar datos y resultados ---
df = pd.read_csv('data/WineQT.csv')
with open('utils/regression_results.json') as f:
	reg_results = json.load(f)
with open('utils/classification_results.json') as f:
	clf_results = json.load(f)

with open('utils/clustering_results.json') as f:
	clu_results = json.load(f)
if 'clusters' in clu_results:
	df['cluster'] = clu_results['clusters']

# --- Tarjeta de introducción ---
st.markdown('<div class="card"><div class="card-title">Presentación</div>'
			'<b>Autor:</b> Daniel Mardones<br>'
			'<b>Proyecto:</b> Visualización y Machine Learning aplicado a la calidad del vino<br>'
			'Esta app integra los principales análisis realizados sobre el popular dataset de Wine Quality, mostrando dashboards interactivos y explicando el uso de las librerías Matplotlib, Seaborn, Plotly y Altair.<br>'
			'</div>', unsafe_allow_html=True)

# --- Dashboards por análisis ---
tab_names = ["Regresión", "Clasificación", "Clustering", "PCA", "Conclusiones"]
# Sincronización de navegación
#if 'active_tab' not in st.session_state:
#	st.session_state['active_tab'] = 0
#selected_tab = st.radio("Ir a sección:", tab_names, index=st.session_state['active_tab'])
#st.session_state['active_tab'] = tab_names.index(selected_tab)
tabs = st.tabs(tab_names)

# --- Regresión ---
with tabs[0]:
	st.markdown(f'''<div class="card"><div class="card-title">Regresión</div>
				Se utiliza regresión lineal para predecir la calidad del vino.<br>
				<b>MSE:</b> {reg_results['mse']:.3f} <b>R2:</b> {reg_results['r2']:.3f}<br>
				Visualizaciones con <b>Matplotlib</b> y <b>Altair</b>.
				</div>''', unsafe_allow_html=True)
	fig, axs = plt.subplots(1, 2, figsize=(12, 5))
	df['quality'].hist(ax=axs[0], bins=10, color='#2980b9')
	axs[0].set_title('Distribución de la calidad del vino')
	axs[1].boxplot(df.select_dtypes(include='number').values)
	axs[1].set_title('Boxplot de variables numéricas')
	axs[1].set_xticklabels(df.select_dtypes(include='number').columns, rotation=90)
	st.pyplot(fig)
	chart = alt.Chart(df).mark_point().encode(
		x='alcohol', y='quality', color='quality', tooltip=['alcohol', 'quality']
	).properties(title='Relación entre Alcohol y Calidad del Vino')
	st.altair_chart(chart, use_container_width=True)
	st.markdown('<div style="background-color:#eaf3fb; border-radius:18px; padding:2rem; margin-top:2rem; color:#154c79; font-size:1.3rem;">Matplotlib es ideal para gráficos estáticos y revisiones iniciales. Altair destaca en visualizaciones estadísticas avanzadas.</div>', unsafe_allow_html=True)

# --- Clasificación ---
with tabs[1]:
	st.markdown(f'''<div class="card"><div class="card-title">Clasificación</div>
				Se utiliza RandomForest para clasificar vinos de alta y baja calidad.<br>
				<b>Accuracy:</b> {clf_results['accuracy']:.3f} <b>F1:</b> {clf_results['f1_score']:.3f}<br>
				Visualizaciones con <b>Seaborn</b> y <b>Plotly</b>.
				</div>''', unsafe_allow_html=True)
	clase_counts = df['quality'].apply(lambda x: 1 if x >= 7 else 0).value_counts().sort_index()
	fig, ax = plt.subplots(figsize=(7,7))
	ax.pie(clase_counts, labels=['Baja calidad','Alta calidad'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
	ax.set_title('Proporción de clases de calidad de vino')
	st.pyplot(fig)
	fig2 = px.scatter(df, x='alcohol', y='quality', color=df['quality'].apply(lambda x: 1 if x >= 7 else 0), title='Alcohol vs Calidad (Clasificación)', labels={'color':'Clase'})
	st.plotly_chart(fig2, use_container_width=True)
	st.markdown('<div style="background-color:#eaf3fb; border-radius:18px; padding:2rem; margin-top:2rem; color:#154c79; font-size:1.3rem;">Seaborn facilita el análisis exploratorio y la visualización de correlaciones. Plotly es potente para dashboards interactivos.</div>', unsafe_allow_html=True)

# --- Clustering ---
with tabs[2]:
	st.markdown('<div class="card"><div class="card-title">Clustering</div>'
				'Se utiliza KMeans para agrupar vinos en clusters.<br>'
				'Visualizaciones con <b>Matplotlib</b> y <b>Plotly</b>.'
				'</div>', unsafe_allow_html=True)
	if 'cluster' not in df.columns:
		st.warning('Ejecuta el notebook de clustering para agregar la columna cluster al dataset.')
	else:
			cluster_counts = df['cluster'].value_counts().sort_index()
			cluster_names = {0: 'Clúster Tradicional', 1: 'Clúster Premium', 2: 'Clúster Innovador'}
			labels = [cluster_names.get(i, f'Cluster {i}') for i in cluster_counts.index]
			colors = plt.cm.Paired.colors
			fig, ax = plt.subplots(figsize=(7,7))
			ax.pie(cluster_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
			ax.set_title('Proporción de clústeres de vino')
			st.pyplot(fig)
			fig2 = px.scatter(df, x='alcohol', y='quality', color='cluster', title='Dispersión de Clústeres (Alcohol vs Calidad)', labels={'cluster':'Clúster'})
			st.plotly_chart(fig2, use_container_width=True)
	st.markdown('<div style="background-color:#eaf3fb; border-radius:18px; padding:2rem; margin-top:2rem; color:#154c79; font-size:1.3rem;">Matplotlib permite visualizar agrupaciones y proporciones. Plotly facilita la exploración interactiva de los clusters.</div>', unsafe_allow_html=True)

# --- PCA ---
with tabs[3]:
	st.markdown('<div class="card"><div class="card-title">Análisis de Componentes Principales (PCA)</div>'
				'Se utiliza PCA para identificar las variables más relevantes y reducir la dimensionalidad.<br>'
				'Visualizaciones con <b>Seaborn</b> y <b>Matplotlib</b>.'
				'</div>', unsafe_allow_html=True)
	try:
		from sklearn.preprocessing import StandardScaler
		from sklearn.decomposition import PCA
		features = df.drop(['quality'], axis=1).select_dtypes(include='number')
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(features)
		pca = PCA(n_components=2)
		principalComponents = pca.fit_transform(X_scaled)
		df_pca = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
		df_pca['quality'] = df['quality']
		fig, ax = plt.subplots(figsize=(10,6))
		sns.scatterplot(x='PC1', y='PC2', hue='quality', data=df_pca, palette='viridis', ax=ax)
		ax.set_title('PCA: Distribución de vinos por calidad')
		st.pyplot(fig)
		importances = pd.DataFrame({'Variable': features.columns, 'PC1': pca.components_[0], 'PC2': pca.components_[1]})
		importances.set_index('Variable', inplace=True)
		fig2, ax2 = plt.subplots(figsize=(12,6))
		importances.plot(kind='bar', ax=ax2)
		ax2.set_title('Importancia de las variables en los componentes principales')
		st.pyplot(fig2)
	except Exception as e:
		st.error(f'Error en el análisis PCA: {e}')
	st.markdown('<div style="background-color:#eaf3fb; border-radius:18px; padding:2rem; margin-top:2rem; color:#154c79; font-size:1.3rem;">Seaborn es excelente para visualizaciones estadísticas y Matplotlib para gráficos de importancia de variables.</div>', unsafe_allow_html=True)

# --- Conclusiones ---
with tabs[4]:
	st.markdown('<div class="card"><div class="card-title">Conclusiones y Estrategias para el Negocio</div>'
				'Las visualizaciones y modelos muestran que variables como el alcohol, la acidez y los sulfitos son clave para la calidad del vino.<br>'
				'</div>', unsafe_allow_html=True)

	st.markdown('<div class="card"><div class="card-title">Dashboard de Segmentaciones y Estrategias</div></div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)
	with col1:
		st.subheader('Segmentación por Calidad')
		high_quality = df[df['quality'] >= 7]
		low_quality = df[df['quality'] < 7]
		st.markdown(f'''<div class="card"><div class="card-title">Vinos de alta calidad</div>
			<h2 style="color:#2980b9;">{len(high_quality)}</h2>
			</div>''', unsafe_allow_html=True)
		st.markdown(f'''<div class="card"><div class="card-title">Vinos de baja calidad</div>
			<h2 style="color:#95a5a6;">{len(low_quality)}</h2>
			</div>''', unsafe_allow_html=True)
		fig, ax = plt.subplots(figsize=(6,4))
		ax.hist([high_quality['alcohol'], low_quality['alcohol']], bins=10, label=['Alta calidad','Baja calidad'], color=['#2980b9','#95a5a6'], alpha=0.7)
		ax.set_title('Alcohol por segmento de calidad')
		ax.set_xlabel('Alcohol')
		ax.set_ylabel('Frecuencia')
		ax.legend()
		st.pyplot(fig)

	with col2:
		st.subheader('Segmentación por Clústeres')
		cluster_names = {0: 'Clúster Tradicional', 1: 'Clúster Premium', 2: 'Clúster Innovador'}
		cluster_counts = df['cluster'].value_counts().sort_index()
		colors = ['#2980b9', '#27ae60', '#8e44ad']
		fig, ax = plt.subplots(figsize=(6,4))
		bars = ax.bar([cluster_names.get(i, str(i)) for i in cluster_counts.index], cluster_counts.values, color=colors)
		ax.set_title('Distribución por Clúster de Vino')
		ax.set_xlabel('Clúster')
		ax.set_ylabel('Cantidad de vinos')
		for bar, count in zip(bars, cluster_counts.values):
			ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count), ha='center', va='bottom', fontsize=12)
		st.pyplot(fig)
		fig2, ax2 = plt.subplots(figsize=(6,4))
		for i, cluster in enumerate(df['cluster'].unique()):
			subset = df[df['cluster'] == cluster]
			ax2.scatter(subset['alcohol'], subset['quality'], label=cluster_names.get(cluster, f'Cluster {cluster}'), alpha=0.6, color=colors[i])
		ax2.set_xlabel('Alcohol')
		ax2.set_ylabel('Calidad')
		ax2.set_title('Dispersión de Clústeres')
		ax2.legend()
		st.pyplot(fig2)

	st.markdown('<div class="card"><div class="card-title">Estrategias Accionables</div>'
				'<ul>'
				'<li><b>Optimización de variables clave:</b> Ajusta los niveles de alcohol, acidez y sulfitos en la producción para maximizar la calidad y satisfacción del cliente.</li>'
				'<li><b>Innovación en productos:</b> Desarrolla nuevas líneas de vino basadas en los insights de PCA y clustering, orientadas a diferentes segmentos de consumidores.</li>'
				'<li><b>Monitoreo continuo:</b> Implementa paneles interactivos para seguimiento en tiempo real de la calidad y variables críticas, facilitando la toma de decisiones ágil y basada en datos.</li>'
				'</ul></div>', unsafe_allow_html=True)

#st.markdown('<div style="background-color:#eaf3fb; border-radius:18px; padding:2rem; margin-top:2rem; color:#154c79; font-size:1.3rem;">Cada librería aporta ventajas: Matplotlib para gráficos estáticos, Seaborn para análisis exploratorio, Plotly para dashboards interactivos y Altair para visualizaciones estadísticas avanzadas.</div>', unsafe_allow_html=True)
