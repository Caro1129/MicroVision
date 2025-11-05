import streamlit as st
import os
import cv2
import numpy as np
import base64
import pdfkit
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import math
import re
from sklearn.decomposition import PCA
from fpdf import FPDF
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
import tempfile
from scipy import stats
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from io import BytesIO
from datetime import datetime


def generar_pdf_reporte_completo():
    # === CONFIGURACIÓN DEL DOCUMENTO ===
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    story = []
    styles = getSampleStyleSheet()

    # === ESTILOS PERSONALIZADOS ===
    estilo_titulo = ParagraphStyle(
        name="Titulo",
        parent=styles["Heading1"],
        fontSize=18,
        alignment=1,  # Centrado
        spaceAfter=20,
        textColor=colors.HexColor("#2E4053")
    )

    estilo_subtitulo = ParagraphStyle(
        name="Subtitulo",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1A5276"),
        spaceBefore=10,
        spaceAfter=10
    )

    estilo_justificado = ParagraphStyle(
        name="Justify",
        parent=styles["Normal"],
        alignment=4,  # Justificado
        leading=15,
        fontSize=11
    )

    story.append(Paragraph("REPORTE DE ANÁLISIS MICROVISION", estilo_titulo))
    fecha = datetime.now().strftime("%d/%m/%Y - %H:%M")
    story.append(Paragraph(f"<b>Fecha de generación:</b> {fecha}", styles["Normal"]))
    story.append(Spacer(1, 1*cm))
    
    def convertir_markdown_a_html(texto):
        # Maneja casos donde el texto no es una cadena (ej. números)
        if not isinstance(texto, str):
            return str(texto)
        # Convierte **texto** a <b>texto</b> usando expresiones regulares
        texto_corregido = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', texto, flags=re.DOTALL)
        return texto_corregido


    # === DESCRIPCIÓN DEL ENSAYO ===
    norma = st.session_state.get("norma", "No especificada")
    microorg = st.session_state.get("microorg_selec", "No especificado")
    medio = st.session_state.get("medio", "medio")
    tiempo = st.session_state.get("tiempo", "tiempo no definido")
    temperatura = st.session_state.get("temperatura", "temperatura no definida")

    descripcion_texto = (
        f"El ensayo se llevó a cabo bajo los lineamientos de la norma {norma}, "
        f"empleando a <i>{microorg}</i> como organismo de ensayo para evaluar la resistencia fúngica del material. "
        f"Las muestras fueron incubadas en {medio.lower()} durante {tiempo} "
        f"a una temperatura de {temperatura} °C, clasificando el grado de biodeterioro conforme a la escala {norma}."
    )

    microorg_descripcion = (
        f"<i>{microorg}</i> es un hongo filamentoso de crecimiento rápido, frecuente en ambientes húmedos y suelos. "
        "Se utiliza como cepa de ensayo en estudios de biodeterioro para determinar la resistencia fúngica de materiales."
    )

    story.append(Paragraph("Descripción del ensayo", estilo_subtitulo))
    story.append(Paragraph(descripcion_texto, estilo_justificado))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(microorg_descripcion, estilo_justificado))
    story.append(Spacer(1, 1*cm))

    # === TABLA DE RESULTADOS (si existe) ===
    if "valores_replicas" in st.session_state and st.session_state["valores_replicas"]:
        story.append(Paragraph("Resultados del análisis", estilo_subtitulo))

        datos_tabla = [["Muestra", "Valor"]] + [
            [f"Replica {i+1}", f"{v:.2f}"] for i, v in enumerate(st.session_state["valores_replicas"])
        ]
        datos_tabla.append(["Media", f"{st.session_state.get('media', 0):.2f}"])
        datos_tabla.append(["Desviación estándar", f"{st.session_state.get('desviacion', 0):.2f}"])

        tabla = Table(datos_tabla, hAlign="CENTER")
        tabla.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D5D8DC")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke)
        ]))
        story.append(tabla)
        story.append(Spacer(1, 1*cm))

    # === CONCLUSIÓN ===
    interpretacion = st.session_state.get("interpretacion", "No se ha generado interpretación para este análisis.")
    
    # FIX: Convertir las marcas de negrita de Markdown (**) a etiquetas ReportLab (<b>)
    # El patrón r'\*\*(.*?)\*\*' encuentra cualquier texto rodeado por ** y lo envuelve en <b>...</b>
    import re
    # === CONCLUSIÓN ===
    interpretacion = st.session_state.get("interpretacion", "No se ha generado interpretación para este análisis.")

    # 🔴 APLICACIÓN DEL FIX: Corrige el formato de la conclusión
    interpretacion_rl = convertir_markdown_a_html(interpretacion)

    story.append(Paragraph("Conclusión", estilo_subtitulo))
    # Usa la variable corregida
    story.append(Paragraph(interpretacion_rl, estilo_justificado))
    story.append(PageBreak())

    # === CONSTRUCCIÓN DEL DOCUMENTO ===
    doc.build(story)
    buffer.seek(0)
    return buffer


def generar_conclusion_texto():
    """Genera texto de conclusión basado en los resultados"""
    norma = st.session_state.get("norma", "")
    microorg = st.session_state.get("microorg_selec", "")
    results = st.session_state.get("results", {})
    num_replicas = st.session_state.get("num_replicas", 0)
    media = st.session_state.get("media", 0)
    desviacion = st.session_state.get("desviacion", 0)
    
    if num_replicas == 1:
        if "AATCC" in norma:
            halo = results.get("inhibition_halo_mm", 0)
            if halo > 0:
                return f"el material evaluado presentó un halo de inhibición de {halo:.2f} mm frente a <i>{microorg}</i>. este resultado indica actividad antibacteriana según la norma {norma}.".lower()
            else:
                return f"no se observó halo de inhibición frente a <i>{microorg}</i>, indicando ausencia de actividad antibacteriana según la norma {norma}.".lower()
        
        elif "ASTM G21" in norma:
            rating = results.get("astm_g21_rating", 0)
            coverage = results.get("coverage_percentage", 0)
            return f"el material obtuvo una calificación de {rating} en la escala astm g21-15, con una cobertura fúngica del {coverage:.2f}%.".lower()
        
        elif "JIS" in norma:
            logR = results.get("log_reduction", 0)
            cumple = "cumple" if logR >= 2 else "no cumple"
            return f"el material presentó una reducción logarítmica de {logR:.2f} frente a <i>{microorg}</i>. por lo tanto, el material {cumple} con la norma {norma}.".lower()
    
    else:  # Múltiples réplicas
        if "ASTM G21" in norma:
            return f"se analizaron {num_replicas} réplicas del ensayo según la norma astm g21-15. el material presentó una cobertura fúngica promedio de {media:.2f}% ± {desviacion:.2f}% (de).".lower()
        
        elif "AATCC" in norma:
            return f"se analizaron {num_replicas} réplicas según la norma {norma}, con un halo de inhibición promedio de {media:.2f} ± {desviacion:.2f} mm (de).".lower()
        
        elif "JIS" in norma:
            return f"el análisis de {num_replicas} réplicas según la norma {norma} mostró una reducción logarítmica promedio de {media:.2f} ± {desviacion:.2f} (de).".lower()
    
    return "no se detectaron resultados válidos para generar una conclusión.".lower()


def agregar_test_t_pdf_tabla(story, test_t_results, temp_files, styles):
    """Agrega la sección de Test T al PDF con tabla y gráfica"""
    
    subtitulo_style = ParagraphStyle(
        'SubtituloCustom',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    story.append(Paragraph("ANÁLISIS ESTADÍSTICO: TEST T DE STUDENT", subtitulo_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Tabla de resultados
    data_ttest = [
        ['Parámetro', 'Valor'],
        ['Estadístico t', f"{test_t_results['t_statistic']:.4f}"],
        ['Valor p', f"{test_t_results['p_value']:.4f}"],
        ['Grados de libertad', str(test_t_results['grados_libertad'])],
        ['Significativo (α=0.05)', 'SÍ' if test_t_results['es_significativo'] else 'NO']
    ]
    
    table_ttest = Table(data_ttest, colWidths=[3*inch, 2*inch])
    table_ttest.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(table_ttest)
    story.append(Spacer(1, 0.2*inch))
    
    # Interpretación
    story.append(Paragraph(f"<b>Interpretación:</b> {test_t_results['interpretacion']}", 
                          styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))

# Ocultar barra superior, menú y pie de página de Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
# CARGAR FONDO DESDE EL MISMO DIRECTORIO 
import os, sys, base64

import base64, streamlit as st

import base64, streamlit as st

def add_bg_from_local(image_file):
    image_path = resource_path(image_file) 
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        
        /* === INICIO DE LA NUEVA MODIFICACIÓN CLAVE === */
        /* Aplicamos margen a la etiqueta <main> que contiene todo el contenido */
        section.main {{
            /* Estos 400px son MUY importantes y son la clave. */
            /* Ponemos un margen izquierdo/derecho que es más ancho que tu franja */
            padding-left: 400px; 
            padding-right: 400px;
        }}
        
        /* Ajustamos el tamaño del contenedor interno por si acaso */
        .block-container {{
            max-width: 100% !important; 
        }}
        /* === FIN DE LA NUEVA MODIFICACIÓN CLAVE === */
        
        </style>
        """,
        unsafe_allow_html=True
    )

class MultiStandardAnalyzer:
    def __init__(self):
        self.standards = {
            'AATCC_TM147': ['Klebsiella pneumoniae', 'Staphylococcus aureus'],
            'ASTM_G21': ['Aspergillus niger', 'Trichoderma viride'],
            'JIS_Z2801': ['Escherichia coli', 'Staphylococcus aureus'],
            'ASTM_E1428': ['Streptomyces species']
        }

    def calibrar_escala_automatica(self, imagen, diametro_real_mm=90):

        # Convertir a escala de grises 
        gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

        # Filtrar ruido y mejorar contraste 
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        gray = cv2.equalizeHist(gray)

        # Detección de bordes 
        edges = cv2.Canny(gray, 30, 120)

        # Detección de círculos con Hough 
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=gray.shape[0] // 4,
            param1=80,
            param2=40,
            minRadius=int(gray.shape[0] * 0.2),
            maxRadius=int(gray.shape[0] * 0.9)
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, radius = circles[0][0]
            diametro_px = radius * 2
            mm_per_pixel = diametro_real_mm / diametro_px

            # Validar rango razonable 
            if not (0.01 < mm_per_pixel < 0.2):
                mm_per_pixel = 0.05  
                return mm_per_pixel, diametro_px  # No mostrar mensaje si está fuera de rango
            else:
                return mm_per_pixel, diametro_px

        else:
            # Si no detecta el círculo, usa valor por defecto
            mm_per_pixel = 0.05
            return mm_per_pixel, None


    def load_and_process_image(self, img_path):
        """Carga y prepara imagen: RGB, PCA (normalizado por canal) y MeanShift sobre RGB suavizado"""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        # Convertir a RGB para mostrar en st.image
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Espacios de color (usar RGB como fuente)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hsl = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        luv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Luv)

        # Concatenar pero SCALEAR cada canal a 0..1 para evitar dominancia numérica
        features = np.concatenate([image_rgb, hsv, hsl, lab, luv], axis=2).astype(np.float32)
        # normalizar canal a canal
        h, w, c = features.shape
        features_reshaped = features.reshape(-1, c)
        # evitar división por 0
        col_min = features_reshaped.min(axis=0)
        col_max = features_reshaped.max(axis=0)
        denom = (col_max - col_min)
        denom[denom == 0] = 1.0
        features_norm = (features_reshaped - col_min) / denom
        features_norm = features_norm.reshape(h, w, c).astype(np.float32)

        pca_img = self.apply_pca(features_norm, n_components=3)

        # MeanShift sobre la imagen RGB suavizada (más estable que sobre PCA en algunos casos)
        rgb_for_ms = (image_rgb.astype(np.uint8)).copy()
        meanshift_img = cv2.pyrMeanShiftFiltering(rgb_for_ms, sp=16, sr=40, maxLevel=1)

        return image_rgb, pca_img, meanshift_img

    def apply_pca(self, features_norm, n_components=3):
        """Aplica PCA a características normalizadas (0..1). Maneja columnas constantes."""
        from sklearn.decomposition import PCA
        h, w, c = features_norm.shape
        X = features_norm.reshape(-1, c).astype(np.float64)

        # Quitar columnas constantes para evitar errores en PCA
        variances = X.var(axis=0)
        non_const_idx = np.where(variances > 1e-8)[0]
        if len(non_const_idx) == 0:
            # todo igual, devolver copia pequeña del RGB
            out = (features_norm[:, :, :3] * 255).astype(np.uint8)
            return out

        X_reduced = X[:, non_const_idx]

        pca = PCA(n_components=min(n_components, X_reduced.shape[1]))
        Xp = pca.fit_transform(X_reduced)

        # Si PCA devolvió menos componentes que n_components, rellenar con ceros
        if Xp.shape[1] < n_components:
            pad = np.zeros((Xp.shape[0], n_components - Xp.shape[1]), dtype=Xp.dtype)
            Xp = np.hstack([Xp, pad])

        # Normalizar cada canal PCA a 0..255
        for i in range(n_components):
            ch = Xp[:, i]
            mn, mx = ch.min(), ch.max()
            if mx - mn <= 1e-9:
                Xp[:, i] = 0
            else:
                Xp[:, i] = (ch - mn) / (mx - mn) * 255.0

        pca_img = Xp.reshape(h, w, n_components).astype(np.uint8)
        return pca_img
    

    
    
    def analyze_halo_TM147_visual_final(self, orig_img, mm_per_pixel=0.05, debug=False):
    
        # --- Asegurar BGR (OpenCV) y eliminar alpha si existe ---
        img = orig_img.copy()
        if img is None:
            raise ValueError("Imagen nula")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Detectar caja de Petri (círculo) ---
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(blur, 60, 150)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                param1=80, param2=30,
                                minRadius=int(min(h, w) * 0.25),
                                maxRadius=int(min(h, w) * 0.48))
        mask_petri = np.zeros_like(gray)
        if circles is not None:
            x, y, r = np.uint16(np.around(circles[0][0]))
            cx_petri, cy_petri, r_petri = int(x), int(y), int(r)
        else:
            cx_petri, cy_petri, r_petri = w // 2, h // 2, int(min(h, w) * 0.45)
        cv2.circle(mask_petri, (cx_petri, cy_petri), r_petri - 5, 255, -1)

        # --- Calibración automática mm/píxel ---
        diametro_real_mm = 90.0
        if r_petri > 0:
            mm_per_pixel = diametro_real_mm / (2 * r_petri)

        # --- Detectar textil (zona central blanca) ---
        # Usamos un umbral alto + morfología para detectar la tela (zona clara)
        _, mask_textil = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        mask_textil = cv2.morphologyEx(mask_textil, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
        contours, _ = cv2.findContours(mask_textil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            (x_t, y_t), r_textil = cv2.minEnclosingCircle(cnt)
            cx_textil, cy_textil, r_textil = int(x_t), int(y_t), int(r_textil)
        else:
            # Fallback razonable
            cx_textil, cy_textil, r_textil = cx_petri, cy_petri, int(r_petri * 0.08)
            cv2.circle(mask_textil, (cx_textil, cy_textil), r_textil, 255, -1)

        # Si el textil detectado es sospechosamente grande, corregir
        if r_textil > r_petri * 0.6:
            r_textil = int(r_petri * 0.1)

        # --- Detección mejorada de crecimiento ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Máscara de verde (ajustable)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Zona anular entre textil y borde para detectar manchas oscuras con umbral adaptativo
        mask_ring = np.zeros_like(gray)
        cv2.circle(mask_ring, (cx_textil, cy_textil), int(r_petri - 6), 255, -1)
        cv2.circle(mask_ring, (cx_textil, cy_textil), int(r_textil + 3), 0, -1)
        ring_pixels = cv2.bitwise_and(gray, gray, mask=mask_ring)

        # Umbral adaptativo en la zona anular (más robusto que un umbral fijo)
        # Convertir a 8-bit si es necesario (ya lo es)
        th_ring = cv2.adaptiveThreshold(ring_pixels, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 7)

        # Combinar detecciones
        mask_growth = cv2.bitwise_or(mask_green, th_ring)

        # Aplicar máscara de Petri y excluir textil
        mask_growth = cv2.bitwise_and(mask_growth, mask_petri)
        mask_growth[mask_textil > 0] = 0

        # Limpiar ruido y filtrar por área
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_growth = cv2.morphologyEx(mask_growth, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_growth = cv2.morphologyEx(mask_growth, cv2.MORPH_OPEN, kernel, iterations=1)

        # Filtrar componentes pequeños y quitar regiones que cubren casi toda la placa
        contours_g, _ = cv2.findContours(mask_growth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_growth_filtered = np.zeros_like(mask_growth)
        petri_area = np.pi * (r_petri ** 2)
        for c in contours_g:
            area = cv2.contourArea(c)
            if area < 50:  # area mínima (ajustable)
                continue
            # Si la región cubre >70% de la placa, descartarla (probablemente segmentación errónea)
            if area > 0.7 * petri_area:
                continue
            cv2.drawContours(mask_growth_filtered, [c], -1, 255, -1)
        mask_growth = mask_growth_filtered

        # --- Calcular halo radial desde borde del textil ---
        halo_mask = np.zeros_like(gray)
        halo_dists = []
        angles_with_growth = 0

        # Precalcular rango radial
        r_min = int(r_textil) + 5
        r_max = int(r_petri) - 6

        if r_min >= r_max:
            # No hay espacio para medir
            if debug:
                print("Advertencia: r_min >= r_max, imposible medir halo.")
        else:
            for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
                found = False
                for r in range(r_min, r_max):
                    x = int(cx_textil + r * np.cos(angle))
                    y = int(cy_textil + r * np.sin(angle))
                    if x < 0 or y < 0 or x >= w or y >= h:
                        break
                    if mask_growth[y, x] > 0:
                        dist_mm = (r - r_textil) * mm_per_pixel
                        halo_dists.append(dist_mm)
                        angles_with_growth += 1
                        # dibujar linea
                        x_start = int(cx_textil + r_textil * np.cos(angle))
                        y_start = int(cy_textil + r_textil * np.sin(angle))
                        cv2.line(halo_mask, (x_start, y_start), (x, y), 255, 1)
                        found = True
                        break
                # Si no encontró crecimiento: no agregar valor (evitar sesgar con máximos)
                # (opcional: podrías añadir un valor máximo, pero solo si la segmentación es fiable)

        avg_halo_mm = float(np.mean(halo_dists)) if len(halo_dists) > 0 else 0.0
        measurements = halo_dists.copy()

        # --- Preparar overlay (colores en BGR) ---
        overlay = img.copy()
        # textil -> rojo claro (BGR)
        overlay[mask_textil > 0] = (60, 60, 220)
        # crecimiento -> verde claro
        overlay[mask_growth > 0] = (60, 220, 60)
        # halo lines -> naranja
        overlay[halo_mask > 0] = (40, 150, 255)

        # crear inhibición (entre textil y crecimiento) usando avg_halo
        if avg_halo_mm > 0:
            radius_inhibition_px = int(r_textil + (avg_halo_mm / mm_per_pixel))
            mask_inhibition = np.zeros_like(gray)
            cv2.circle(mask_inhibition, (cx_textil, cy_textil), radius_inhibition_px, 255, -1)
            mask_inhibition[mask_textil > 0] = 0
            mask_inhibition[mask_growth > 0] = 0
            # amarillo claro
            overlay[mask_inhibition > 0] = (150, 255, 255)

        # Mezclar
        overlay_final = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        overlay_final = cv2.bitwise_and(overlay_final, overlay_final, mask=mask_petri)

        # Dibujar referencia textil
        cv2.circle(overlay_final, (cx_textil, cy_textil), int(r_textil), (0, 0, 255), 2)  # rojo BGR

        # Texto
        if avg_halo_mm > 0:
            text = f"Halo: {avg_halo_mm:.2f} mm"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay_final, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
            cv2.putText(overlay_final, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if debug:
            print(f"mm_per_pixel = {mm_per_pixel:.6f}")
            print(f"r_textil = {r_textil}px, r_petri = {r_petri}px")
            print(f"Ángulos con crecimiento: {angles_with_growth}/360")
            print(f"Mediciones: {len(measurements)} promedio = {avg_halo_mm:.2f} mm")
            # Retornar máscaras para debugging visual (puedes guardarlas o mostrarlas)
            return mask_textil, mask_growth, avg_halo_mm, overlay_final, measurements, (cx_textil, cy_textil, r_textil), {
                'mask_petri': mask_petri, 'mask_inhibition': (mask_inhibition if avg_halo_mm>0 else np.zeros_like(gray)),
                'halo_mask': halo_mask
            }

        # return estándar (6 valores)
        return mask_textil, mask_growth, avg_halo_mm, overlay_final, measurements, (cx_textil, cy_textil, r_textil)



    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    def count_colonies_opencv(self, original_img, segmentacion=None, debug=False):
        """
        Conteo de colonias optimizado para placas DENSAS mediante el algoritmo Watershed.
        Corrección: Se asegura que el Watershed y el dibujo se hagan sobre una imagen BGR.
        """
        
        # --- 1) Preparación y Enmascaramiento ---
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Detección de la placa Petri
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min(h,w)//4, param1=50, param2=30,
            minRadius=int(min(h,w)*0.25), maxRadius=int(min(h,w)*0.55)
        )

        mask = np.zeros_like(gray)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            (x, y, r) = circles[0, 0] 
            cv2.circle(mask, (x, y), r, (255), thickness=-1)
            gray_processed = cv2.bitwise_and(gray, gray, mask=mask)
        else:
            gray_processed = gray 


        # --- 2) Umbralización Adaptativa y Morfología ---
        block_size = 61
        C_value = 10
        
        thresh = cv2.adaptiveThreshold(
            gray_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, C_value
        ) 

        # Operación de Apertura: 
        kernel = np.ones((3, 3), np.uint8) 
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) # Iterations=2

        
        # --- 3) Preparación de Marcadores para Watershed ---

        # A. Fondo (Background)
        sure_bg = cv2.dilate(opened, kernel, iterations=3) 

        # B. Primer Plano (Foreground - centro seguro de colonia)
        dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

        # AJUSTE FINO: Reducimos el umbral a 0.2 para máxima sensibilidad
        ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0) 
        sure_fg = np.uint8(sure_fg)

        # C. Región Desconocida
        unknown = cv2.subtract(sure_bg, sure_fg)

        
        # --- 4) Watershed y Conteo Final ---
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1 
        markers[unknown == 255] = 0
        
        # 💥 CORRECCIÓN CRÍTICA: Base BGR para el Watershed
        # Usamos la imagen RGB original para que el Watershed pueda dibujar las fronteras correctamente.
        base_img_watershed = original_img.copy() 
        markers = cv2.watershed(base_img_watershed, markers) 

        # Contar
        colonies_count = np.max(markers) - 1 

        # --- 5) Resultados y Debug ---
        
        # El dibujo se hace sobre una copia de la imagen original
        detected_img = original_img.copy()
        
        # Dibujar las líneas divisorias (fronteras de Watershed) en la imagen
        detected_img[markers == -1] = [0, 255, 0] # El color Verde

        if debug:
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            axes[0].imshow(gray_processed, cmap='gray'); axes[0].set_title('1. Imagen Enmascarada')
            axes[1].imshow(opened, cmap='gray'); axes[1].set_title('2. Umbral Limpio (Apertura)')
            axes[2].imshow(dist_transform, cmap='jet'); axes[2].set_title(f'3. Transformada de Distancia (Umbral 0.2)')
            axes[3].imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)); axes[3].set_title(f'4. Watershed | Count: {colonies_count}')
            plt.show()

        return colonies_count, detected_img











    # ========== FUNCIÓN AUXILIAR PARA VISUALIZAR ==========
    def visualizar_colonias_JIS(self, original_img, colonies, count):
        """
        Crea una visualización colorida de las colonias detectadas
        """
        # Imagen base
        result = original_img.copy()
        
        # Crear overlay translúcido
        overlay = np.zeros_like(result)
        
        # Colorear cada colonia
        for i, colony in enumerate(colonies):
            # Color verde brillante con transparencia
            cv2.drawContours(overlay, [colony], -1, (0, 255, 100), -1)
            
            # Borde rojo grueso
            cv2.drawContours(result, [colony], -1, (255, 0, 0), 3)
            
            # Numerar colonias (opcional)
            M = cv2.moments(colony)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    result, 
                    str(i+1), 
                    (cx-10, cy+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 0),  # Amarillo
                    2
                )
        
        # Mezclar overlay
        if len(colonies) > 0:
            result = cv2.addWeighted(result, 0.65, overlay, 0.35, 0)
        
        # Agregar contador principal
        cv2.putText(
            result,
            f"Colonias detectadas: {count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            4
        )
        
        return result
    






    def analyze_fungal_growth(self, original_img, segmented_img, microorganism=None):
        """
        Análisis ASTM G21-15 - Aspergillus niger ESTRICTO
        Solo detecta crecimiento NEGRO/MARRÓN OSCURO real
        """
        fungal_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        if microorganism:
            microorganism = microorganism.lower()
        else:
            microorganism = ""

        es_aspergillus = "aspergillus" in microorganism or "niger" in microorganism
        es_trichoderma = "trichoderma" in microorganism or "viride" in microorganism or "virens" in microorganism
        
        # ✅ Inicialización por defecto para evitar el error
        useful_area = h * w  # o np.sum(gray > 0) si prefieres el área no nula

        # === Detectar y excluir la muestra central ===
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        white_mask_cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_large)
        specimen_mask = np.zeros_like(gray)
        contours, _ = cv2.findContours(white_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(specimen_mask, [largest_contour], -1, 255, -1)
            specimen_mask = cv2.dilate(specimen_mask, kernel_large, iterations=1)

        if es_aspergillus:
            print("🟤 Detectando Aspergillus niger (SOLO zonas NEGRAS)...")

            # --- 0) Espacios de color ---
            hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
            hue, sat, val = cv2.split(hsv)
            lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0].astype(np.uint8)
            b = lab[:, :, 2]  # ← AGREGADO: necesario para detectar tela

            # --- 1) Detectar Petri ---
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blur, 30, 100)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                    param1=50, param2=30,
                                    minRadius=int(min(h, w) * 0.3),
                                    maxRadius=int(min(h, w) * 0.48))

            petri_mask = np.zeros_like(gray, dtype=np.uint8)
            petri_center = None
            petri_radius = None
            
            if circles is not None:
                x, y, r = np.uint16(np.around(circles[0, 0]))
                petri_center = (x, y)
                petri_radius = r
                cv2.circle(petri_mask, (x, y), int(r * 0.88), 255, -1)
                print(f"✅ Petri: ({x},{y}), r={int(r*0.88)}px")
            else:
                cy, cx = h // 2, w // 2
                safe_radius = int(min(h, w) * 0.42)
                cv2.circle(petri_mask, (cx, cy), safe_radius, 255, -1)

            # --- 2) Excluir borde del Petri ---
            border_mask = np.zeros_like(gray)
            if petri_center and petri_radius:
                outer = np.zeros_like(gray)
                inner = np.zeros_like(gray)
                cv2.circle(outer, petri_center, petri_radius - 5, 255, -1)
                cv2.circle(inner, petri_center, int(petri_radius * 0.88), 255, -1)
                border_mask = cv2.subtract(outer, inner)

            # --- 3) Máscara de análisis ---
            analysis_mask = cv2.bitwise_and(petri_mask, cv2.bitwise_not(specimen_mask))
            if np.sum(analysis_mask) == 0:
                analysis_mask = petri_mask.copy()
            
            # --- 3b) 🧵 EXCLUIR TELA EXTERNA ---
            # NUEVO: Crear textile_mask para Aspergillus
            high_L = cv2.inRange(L, 170, 255)
            low_S = cv2.inRange(sat, 0, 70)
            beige_range = cv2.inRange(b, 120, 160)
            textile_mask = cv2.bitwise_and(cv2.bitwise_and(high_L, low_S), beige_range)
            
            # Suavizar
            kernel_textile = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            textile_mask = cv2.morphologyEx(textile_mask, cv2.MORPH_CLOSE, kernel_textile, iterations=2)
            
            # Crear zona segura: solo interior del Petri
            if petri_center and petri_radius:
                safe_zone = np.zeros_like(gray)
                cv2.circle(safe_zone, petri_center, int(petri_radius * 0.92), 255, -1)
                analysis_mask = cv2.bitwise_and(analysis_mask, safe_zone)
                # Excluir también la tela de la máscara de análisis
                analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(textile_mask))
                print(f"✅ Restringido a zona segura del Petri y excluida tela")

            # --- 4) Preprocesamiento suave ---
            spec_mask = cv2.inRange(val, 240, 255)
            if np.sum(spec_mask) > 50:
                inpaint_rgb = cv2.inpaint(original_img, spec_mask, 3, cv2.INPAINT_TELEA)
                gray_proc = cv2.cvtColor(inpaint_rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray_proc = gray.copy()

            # CLAHE suave
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            gray_clahe = clahe.apply(gray_proc)

            # --- 5) 🔥 ESTRATEGIA: Detectar zonas REALMENTE oscuras ---
            masked_pixels = gray_clahe[analysis_mask > 0]
            
            if masked_pixels.size > 0:
                p5 = np.percentile(masked_pixels, 5)
                p10 = np.percentile(masked_pixels, 10)
                p15 = np.percentile(masked_pixels, 15)
                
                # 🎯 UMBRAL MÁS ESTRICTO: solo lo MÁS oscuro
                if p5 < 50:  # Hay zonas REALMENTE negras
                    dark_thr = int(p15 * 0.95)  # Usar percentil 15 más bajo
                elif p10 < 70:  # Zonas oscuras
                    dark_thr = int(p15 * 1.0)
                else:  # No hay zonas muy oscuras
                    dark_thr = int(p10 * 1.1)
                
                # Limitar a rango más estricto
                dark_thr = np.clip(dark_thr, 50, 90)  # ← MÁS ESTRICTO
                print(f"🎯 Umbral ESTRICTO: {dark_thr} (p5={p5:.1f}, p10={p10:.1f}, p15={p15:.1f})")
            else:
                dark_thr = 70

            # --- 6) Máscara principal: SOLO zonas MUY OSCURAS ---
            _, dark_mask = cv2.threshold(gray_clahe, dark_thr, 255, cv2.THRESH_BINARY_INV)
            dark_mask = cv2.bitwise_and(dark_mask, analysis_mask)
            
            # 🔥 NUEVO: Excluir zonas amarillas/beige del MEDIO
            yellow_beige = cv2.inRange(hue, 10, 40)  # Tonos amarillo-beige
            medium_bright = cv2.inRange(val, 100, 200)  # Brillo medio-alto
            yellow_zone = cv2.bitwise_and(yellow_beige, medium_bright)
            dark_mask[yellow_zone > 0] = 0  # Eliminar zonas amarillas

            # --- 7) 🔥 FILTRO CRÍTICO: Eliminar SOLO zonas beige MUY CLARAS ---
            beige_tones = cv2.inRange(hue, 8, 38)
            very_bright = cv2.inRange(val, 130, 255)
            low_sat = cv2.inRange(sat, 0, 85)
            
            beige_mask = cv2.bitwise_and(
                cv2.bitwise_and(beige_tones, very_bright),
                low_sat
            )
            
            kernel_beige = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            beige_mask = cv2.morphologyEx(beige_mask, cv2.MORPH_CLOSE, kernel_beige, iterations=1)
            
            print(f"🚫 Zona beige MUY clara: {np.sum(beige_mask>0)} px")

            # --- 8) 🎯 FILTRO ADICIONAL: Validar con saturación ---
            valid_dark = dark_mask.copy()
            beige_and_bright = cv2.bitwise_and(beige_mask, cv2.inRange(gray_clahe, 110, 255))
            valid_dark[beige_and_bright > 0] = 0

            # --- 9) Textura como VALIDACIÓN ---
            ksz = 7
            kernel = np.ones((ksz, ksz), np.float32) / (ksz**2)
            mean_local = cv2.filter2D(gray_clahe.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray_clahe.astype(np.float32))**2, -1, kernel)
            variance = sqr_mean - mean_local**2
            variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            _, texture_mask = cv2.threshold(variance_norm, 12, 255, cv2.THRESH_BINARY)

            # --- 10) Combinación CONSERVADORA ---
            combined = (
                valid_dark.astype(np.float32) * 0.75 +
                texture_mask.astype(np.float32) * 0.25
            )
            combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, fungal_mask = cv2.threshold(combined, 100, 255, cv2.THRESH_BINARY)

            # --- 11) Limpieza morfológica MUY SUAVE ---
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
            fungal_mask = cv2.morphologyEx(fungal_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            fungal_mask = cv2.morphologyEx(fungal_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

            # --- 12) Excluir borde y beige claro ---
            # CORREGIDO: Ya no usamos textile_mask aquí porque ya fue excluida de analysis_mask
            if np.sum(border_mask) > 0:
                fungal_mask[border_mask > 0] = 0
            
            fungal_mask[beige_and_bright > 0] = 0
            fungal_mask = cv2.bitwise_and(fungal_mask, analysis_mask)

            # --- 13) Eliminar regiones pequeñas ---
            contours, _ = cv2.findContours(fungal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 25
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    cv2.drawContours(fungal_mask, [cnt], -1, 0, -1)

            # --- 14) 🔥 VALIDACIÓN FINAL ---
            final_contours, _ = cv2.findContours(fungal_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            verified_mask = np.zeros_like(fungal_mask)
            
            for cnt in final_contours:
                temp_mask = np.zeros_like(fungal_mask)
                cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
                
                region_pixels = gray_clahe[temp_mask > 0]
                if len(region_pixels) > 0:
                    region_mean = np.mean(region_pixels)
                    region_min = np.min(region_pixels)
                    if region_mean < 118 or region_min < 85:
                        cv2.drawContours(verified_mask, [cnt], -1, 255, -1)

            fungal_mask = verified_mask
            print(f"✅ Píxeles finales (verificados): {np.sum(fungal_mask>0)}")
            useful_area = np.sum(analysis_mask > 0)

        elif es_trichoderma:
            print("🟢 Detectando Trichoderma spp. (micelio blanco + esporulación verde)...")

            # === 1️⃣ Conversión de espacios de color ===
            hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            hue, sat, val = cv2.split(hsv)
            L = lab[:, :, 0]
            a = lab[:, :, 1]
            b = lab[:, :, 2]
            h, w = gray.shape

            # === 2️⃣ Detectar caja de Petri ===
            gray_blur = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(
                gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                param1=80, param2=40, minRadius=100, maxRadius=int(min(h, w)/2)
            )
            petri_mask = np.zeros_like(gray)
            if circles is not None:
                x, y, r = np.uint16(np.around(circles[0, 0]))
                cv2.circle(petri_mask, (x, y), r - 8, 255, -1)
                print(f"✅ Caja de Petri detectada: centro=({x},{y}), radio={r}px")
            else:
                petri_mask[:] = 255
                print("⚠️ No se detectó círculo, usando máscara completa.")

            analysis_mask = cv2.bitwise_and(petri_mask, cv2.bitwise_not(specimen_mask))

            # === 3️⃣ Detectar tela y eliminarla ===
            high_L = cv2.inRange(L, 175, 255)          # Muy clara
            low_S = cv2.inRange(sat, 0, 80)            # Casi sin color
            neutral_a = cv2.inRange(a, 120, 145)       # Sin tono rojizo
            neutral_b = cv2.inRange(b, 120, 155)       # Sin tono amarillo ni azul

            textile_mask = cv2.bitwise_and(high_L, low_S)
            textile_mask = cv2.bitwise_and(textile_mask, neutral_a)
            textile_mask = cv2.bitwise_and(textile_mask, neutral_b)

            kernel_textile = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            textile_mask = cv2.morphologyEx(textile_mask, cv2.MORPH_CLOSE, kernel_textile, iterations=2)
            textile_mask = cv2.dilate(textile_mask, kernel_textile, iterations=1)

            # === 4️⃣ Generar mapa de textura (micelio real) ===
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = cv2.convertScaleAbs(laplacian)
            texture_mask = cv2.inRange(laplacian_abs, 8, 255)

            # Reagregar textura solo FUERA de la tela
            texture_outside = cv2.bitwise_and(texture_mask, cv2.bitwise_not(textile_mask))
            analysis_mask = cv2.bitwise_or(analysis_mask, texture_outside)

            # Asegurar que la tela se excluya completamente
            analysis_mask[textile_mask > 0] = 0

            # === 5️⃣ Micelio blanco ===
            white_zone = cv2.inRange(val, 130, 240)
            low_sat = cv2.inRange(sat, 0, 90)
            medium_L = cv2.inRange(L, 60, 200)
            white_mask = cv2.bitwise_and(cv2.bitwise_and(white_zone, low_sat), medium_L)
            white_mask = cv2.bitwise_and(white_mask, analysis_mask)

            # === 6️⃣ Esporulación verde ===
            green_h = cv2.inRange(hue, 35, 85)
            green_s = cv2.inRange(sat, 70, 255)
            green_v = cv2.inRange(val, 80, 235)
            green_mask = cv2.bitwise_and(cv2.bitwise_and(green_h, green_s), green_v)
            green_mask = cv2.bitwise_and(green_mask, analysis_mask)

            # Excluir tela de ambos
            white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(textile_mask))
            green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(textile_mask))

            # === 7️⃣ Fusión y limpieza ===
            fungal_mask = cv2.bitwise_or(white_mask, green_mask)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            fungal_mask = cv2.morphologyEx(fungal_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            fungal_mask = cv2.morphologyEx(fungal_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            fungal_mask[petri_mask == 0] = 0

            # === 8️⃣ Resultado final ===
            fungal_mask[textile_mask > 0] = 0
            final_count = np.sum(fungal_mask > 0)
            useful_area = np.sum(analysis_mask > 0)
            coverage_percentage = (final_count / useful_area * 100) if useful_area > 0 else 0
            print(f"✅ Píxeles detectados: {final_count} ({coverage_percentage:.2f}% del área útil)")

        # =====================================================================
        # 📈 Cálculo ASTM G21-15
        # =====================================================================
        fungal_pixels = np.sum(fungal_mask > 0)
        coverage = (fungal_pixels / useful_area * 100) if useful_area > 0 else 0

        # Escala ASTM
        if coverage == 0:
            rating = 0
        elif coverage < 10:
            rating = 1
        elif coverage <= 30:
            rating = 2
        elif coverage <= 60:
            rating = 3
        else:
            rating = 4

        print(f"📊 Cobertura: {coverage:.2f}% | Rating ASTM: {rating}/4")
        print(f"🧩 fungal_pixels={fungal_pixels}, useful_area={useful_area}")

        return rating, coverage, fungal_mask
    
    
    
    def analyze_streptomyces_growth(self, original_img, segmented_img):
        """
        Análisis ASTM E1428-24 - Streptomyces spp.
        Detección basada en textura filamentosa.
        Solo se colorea el micelio; textil central y borde de placa se excluyen completamente.
        """

        # --- 0️⃣ Preparación ---
        if original_img.shape[2] == 3:
            img_rgb = original_img.copy()
        else:
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, w = gray.shape

        # --- 1️⃣ Detección de placa Petri ---
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=80, param2=40,
            minRadius=int(min(h, w) * 0.28),
            maxRadius=int(min(h, w) / 2)
        )

        petri_mask = np.zeros_like(gray, dtype=np.uint8)
        center_y, center_x = h // 2, w // 2
        radius = int(min(h, w) * 0.45)

        if circles is not None:
            x, y, r = np.uint16(np.around(circles[0, 0]))
            center_x, center_y, radius = x, y, int(r * 0.9)
            cv2.circle(petri_mask, (center_x, center_y), radius, 255, -1)
        else:
            cv2.circle(petri_mask, (center_x, center_y), radius, 255, -1)

        # --- 2️⃣ Detectar textil central (MEJORADO) ---
        gray_f = gray.astype(float)
        mean_large = cv2.blur(gray_f, (15, 15))
        sqr_large = cv2.blur(gray_f ** 2, (15, 15))
        variance_large = sqr_large - (mean_large ** 2)
        std_dev_large = np.sqrt(np.abs(variance_large))

        # Textil: std_dev muy bajo (umbral aumentado)
        textile_mask = cv2.inRange(std_dev_large.astype(np.uint8), 0, 5)

        # Limitar a área central más grande
        inner_radius = int(radius * 0.45)
        center_mask = np.zeros_like(gray)
        cv2.circle(center_mask, (center_x, center_y), inner_radius, 255, -1)
        textile_mask = cv2.bitwise_and(textile_mask, center_mask)

        # Detectar áreas blancas (textil por color)
        white_textile = cv2.inRange(gray, 180, 255)
        textile_mask = cv2.bitwise_or(textile_mask, white_textile)

        # Expandir máscara del textil
        kernel_textile = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        textile_mask = cv2.dilate(textile_mask, kernel_textile, iterations=2)
        # --- 3️⃣ Detectar micelio por textura ---
        mean_small = cv2.blur(gray_f, (7, 7))
        sqr_small = cv2.blur(gray_f ** 2, (7, 7))
        variance_small = sqr_small - (mean_small ** 2)
        std_dev = np.sqrt(np.abs(variance_small))

        micelio_mask = cv2.inRange(std_dev.astype(np.uint8), 3, 40)
        # Limitar a placa
        micelio_mask = cv2.bitwise_and(micelio_mask, petri_mask)
        # Excluir textil
        fungal_mask = cv2.bitwise_and(micelio_mask, cv2.bitwise_not(textile_mask))

        # --- 4️⃣ Limpieza morfológica ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fungal_mask = cv2.morphologyEx(fungal_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fungal_mask = cv2.morphologyEx(fungal_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- 5️⃣ Excluir reflejos y blancos ---
        val = hsv[:, :, 2]
        bright = cv2.inRange(val, 240, 255)
        white_pure = cv2.inRange(gray, 200, 255)
        fungal_mask[bright > 0] = 0
        fungal_mask[white_pure > 0] = 0

        # --- 6️⃣ Borde externo adaptativo ---
        erosion_size = max(5, int(radius * 0.08))
        safe_petri = cv2.erode(petri_mask, np.ones((erosion_size, erosion_size), np.uint8))
        fungal_mask = cv2.bitwise_and(fungal_mask, safe_petri)

        # --- 7️⃣ Filtrado por componentes ---
        useful_area = np.count_nonzero(safe_petri)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fungal_mask, connectivity=8)

        min_area = max(50, int(useful_area * 0.002))
        max_area = int(useful_area * 0.20)

        cleaned_mask = np.zeros_like(fungal_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                cleaned_mask[labels == i] = 255

        fungal_mask = cleaned_mask

        # --- 8️⃣ Cobertura ---
        fungal_pixels = np.count_nonzero(fungal_mask)
        coverage = (fungal_pixels / useful_area) * 100 if useful_area > 0 else 0
        coverage = np.clip(coverage, 0, 100)
        has_growth = coverage > 2.0

        # --- 9️⃣ Visualización SOLO micelio ---
        overlay = np.zeros_like(img_rgb)
        overlay[fungal_mask > 0] = [255, 180, 0]  # naranja-amarillo para micelio
        colored_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

        # --- DEBUG ---
        print(f"🔬 Streptomyces (solo micelio)")
        print(f"   std_dev: {np.min(std_dev):.2f}-{np.max(std_dev):.2f}")
        print(f"   Texture px: {np.count_nonzero(micelio_mask)}")
        print(f"   Textil excluido px: {np.count_nonzero(textile_mask)}")
        print(f"   Final mask: {fungal_pixels} ({coverage:.2f}%)")
        print(f"📊 Crecimiento visible: {has_growth}")

        return coverage, colored_img, has_growth










    def _map_standard(self, standard):
        """Normaliza nombre de norma"""
        if not standard:
            return None
        s = standard.lower().replace(" ", "").replace("-", "").replace("_", "")
        
        if "aatcc" in s or "tm147" in s:
            return "AATCC_TM147"
        if "g21" in s or "astmg21" in s:
            return "ASTM_G21"
        if "jis" in s or "z2801" in s:
            return "JIS_Z2801"
        if "e1428" in s or "astme1428" in s:
            return "ASTM_E1428"
        
        return standard
        
    def get_g21_interpretation(self, rating):
        interpretations = {
            0: "Sin crecimiento - Excelente resistencia",
            1: "Trazas de crecimiento (< 10%) - Buena resistencia", 
            2: "Crecimiento ligero (10-30%) - Resistencia moderada",
            3: "Crecimiento moderado (30-60%) - Resistencia limitada",
            4: "Crecimiento severo (> 60%) - Sin resistencia"
        }
        return interpretations.get(rating, "Rating no válido")
    
    def get_jis_interpretation(self, log_reduction):
        if log_reduction >= 3.0:
            return "Excelente actividad antimicrobiana (≥3 log)"
        elif log_reduction >= 2.0:
            return "Buena actividad antimicrobiana (≥2 log)"
        elif log_reduction >= 1.0:
            return "Actividad antimicrobiana moderada (≥1 log)"
        else:
            return "Actividad antimicrobiana insuficiente (<1 log)"
        
    def analyze_by_standard(self, img_path, microorganism, standard, mm_per_pixel, control_count=None, control_path=None):
        """Análisis según norma especificada"""
        standard_key = self._map_standard(standard)
        if not standard_key:
            raise ValueError("Norma no especificada")
        
        print(f" Analizando: {microorganism}")
        print(f" Norma: {standard_key}")
        
        # Procesar imagen de control si existe
        control_img_rgb = None
        control_pca = None
        control_meanshift = None
        control_processed = None
        actual_control_count = None
        
        if control_path:
            print(f" Procesando imagen de CONTROL...")
            control_img_rgb, control_pca, control_meanshift = self.load_and_process_image(control_path)
            actual_control_count, control_colonies = self.count_colonies_opencv(control_img_rgb, control_meanshift)
            
            # Crear imagen procesada del control con contornos verdes
            control_processed = control_img_rgb.copy()
            cv2.drawContours(control_processed, control_colonies, -1, (0, 255, 0), 2)
            print(f"  Control: {actual_control_count} colonias detectadas")

        # Procesar imagen tratada
        print(f"Procesando imagen TRATADA...")
        original_img, pca_img, meanshift_img = self.load_and_process_image(img_path)
        
        results = {
            'microorganism': microorganism,
            'standard': standard_key,
            'image_path': img_path
        }
        
        # --- AATCC TM147 ---
        # --- AATCC TM147 ---
        if 'AATCC' in standard_key or 'TM147' in standard_key:
            print("🔍 Analizando halo de inhibición (AATCC TM147)...")

            # ⭐ CORRECCIÓN: Capturar los 6 valores que retorna la función
            mask_textil, mask_microbio, avg_halo, overlay_img, measurements, halo_center = \
                self.analyze_halo_TM147_visual_final(original_img, mm_per_pixel)

            # Actualizar resultados
            results.update({
                'inhibition_halo_mm': round(avg_halo, 2),
                'has_inhibition': avg_halo > 0,
                'interpretation': 'Efectivo' if avg_halo > 1.0 else 'No efectivo'
            })
    
            
        elif 'ASTM_G21' in standard_key or 'G21' in standard_key:
            rating, coverage_percentage, fungal_mask = self.analyze_fungal_growth(
                original_img, meanshift_img, microorganism=microorganism  # ← NUEVO
            )
            
            results.update({
                'astm_g21_rating': rating,
                'coverage_percentage': round(coverage_percentage, 2),
                'interpretation': self.get_g21_interpretation(rating)
            })
            
        elif 'JIS' in standard_key or 'Z2801' in standard_key:
            # Contar colonias en la imagen tratada
            treated_count, treated_colonies = self.count_colonies_opencv(original_img, meanshift_img)
            print(f"   ✓ Tratada: {treated_count} colonias detectadas")
            
            # Calcular reducción logarítmica
            log_reduction = None
            if actual_control_count is not None and actual_control_count > 0:
                # Evitar división por cero usando max(1, x)
                treated_for_calc = max(1, treated_count)
                control_for_calc = max(1, actual_control_count)
                log_reduction = math.log10(control_for_calc) - math.log10(treated_for_calc)
                print(f"    Reducción log: {log_reduction:.2f}")
                print(f"    Control={control_for_calc}, Tratada={treated_for_calc}")
            else:
                print("    No se pudo calcular reducción log: control count inválido")
            
            results.update({
                'treated_count': treated_count,
                'control_count': actual_control_count if actual_control_count is not None else 'No disponible',
                'log_reduction': round(log_reduction, 2) if log_reduction is not None else 'No calculable',
                'interpretation': self.get_jis_interpretation(log_reduction) if log_reduction is not None else 'Datos incompletos'
            })
            
        elif 'ASTM_E1428' in standard_key or 'E1428' in standard_key:
            coverage_percentage, fungal_mask, has_growth = self.analyze_streptomyces_growth(
                original_img, meanshift_img
            )
            results.update({
                'coverage_percentage': round(coverage_percentage, 2),
                'has_visible_growth': has_growth,
                'material_resistance': 'No resistente' if has_growth else 'Resistente',
                'interpretation': 'Presencia de crecimiento' if has_growth else 'Ausencia de crecimiento'
            })



        
        return results, original_img, pca_img, meanshift_img, control_img_rgb, control_pca, control_meanshift, control_processed

def realizar_test_t_flexible(valores_grupo1, valores_grupo2, nombre_grupo1="Control", nombre_grupo2="Tratada", alpha=0.05):
    
    # CASO 1: Menos de 1 réplica en algún grupo
    if len(valores_grupo1) < 1 or len(valores_grupo2) < 1:
        return {
            'error': f'Se necesita al menos 1 réplica en cada grupo (Grupo 1: {len(valores_grupo1)}, Grupo 2: {len(valores_grupo2)})',
            'suficientes_datos': False,
            'tipo_analisis': 'ninguno'
        }
    
    # Convertir a arrays numpy
    grupo1_array = np.array(valores_grupo1)
    grupo2_array = np.array(valores_grupo2)
    
    # Estadísticas descriptivas básicas
    media_grupo1 = np.mean(grupo1_array)
    media_grupo2 = np.mean(grupo2_array)
    n1 = len(valores_grupo1)
    n2 = len(valores_grupo2)
    
    # CASO 2: Solo 1 réplica en cada grupo - COMPARACIÓN SIMPLE
    if n1 == 1 and n2 == 1:
        diff_means = media_grupo1 - media_grupo2
        porcentaje_reduccion = ((media_grupo1 - media_grupo2) / media_grupo1 * 100) if media_grupo1 != 0 else 0
        
        return {
            'suficientes_datos': True,
            'tipo_analisis': 'simple',
            'media_grupo1': media_grupo1,
            'media_grupo2': media_grupo2,
            'n_grupo1': n1,
            'n_grupo2': n2,
            'nombre_grupo1': nombre_grupo1,
            'nombre_grupo2': nombre_grupo2,
            'diferencia_medias': diff_means,
            'porcentaje_reduccion': porcentaje_reduccion,
            'interpretacion': f"Con 1 réplica por grupo: {nombre_grupo1} = {media_grupo1:.2f}, {nombre_grupo2} = {media_grupo2:.2f}. "
                            f"Diferencia observada: {diff_means:.2f} ({porcentaje_reduccion:.1f}% de reducción). "
                            f"NOTA: No se puede realizar Test T estadístico con n=1. Se requieren al menos 2 réplicas por grupo para análisis estadístico robusto.",
            'advertencia': 'Se necesitan al menos 2 réplicas en cada grupo para realizar Test T estadístico'
        }
    
    # CASO 3: Menos de 2 réplicas en al menos un grupo - COMPARACIÓN LIMITADA
    if n1 < 2 or n2 < 2:
        # Calcular desviación solo para el grupo que tiene múltiples réplicas
        std_grupo1 = np.std(grupo1_array, ddof=1) if n1 > 1 else None
        std_grupo2 = np.std(grupo2_array, ddof=1) if n2 > 1 else None
        diff_means = media_grupo1 - media_grupo2
        porcentaje_reduccion = ((media_grupo1 - media_grupo2) / media_grupo1 * 100) if media_grupo1 != 0 else 0
        
        return {
            'suficientes_datos': True,
            'tipo_analisis': 'limitado',
            'media_grupo1': media_grupo1,
            'media_grupo2': media_grupo2,
            'std_grupo1': std_grupo1,
            'std_grupo2': std_grupo2,
            'n_grupo1': n1,
            'n_grupo2': n2,
            'nombre_grupo1': nombre_grupo1,
            'nombre_grupo2': nombre_grupo2,
            'diferencia_medias': diff_means,
            'porcentaje_reduccion': porcentaje_reduccion,
            'interpretacion': f"{nombre_grupo1}: {media_grupo1:.2f} (n={n1}), {nombre_grupo2}: {media_grupo2:.2f} (n={n2}). "
                            f"Diferencia: {diff_means:.2f} ({porcentaje_reduccion:.1f}% de reducción). "
                            f" Test T no disponible: se requieren al menos 2 réplicas en ambos grupos.",
            'advertencia': f'Grupo con 1 sola réplica: {nombre_grupo1 if n1 == 1 else nombre_grupo2}. Se requieren ≥2 réplicas en ambos grupos para Test T.'
        }
    
    # CASO 4: Al menos 2 réplicas en cada grupo - TEST T COMPLETO
    std_grupo1 = np.std(grupo1_array, ddof=1)
    std_grupo2 = np.std(grupo2_array, ddof=1)
    
    # Realizar Test T para muestras independientes
    t_statistic, p_value = stats.ttest_ind(grupo1_array, grupo2_array)
    
    # Calcular grados de libertad
    df = n1 + n2 - 2
    
    # Determinar si es significativo
    es_significativo = p_value < alpha
    
    # Calcular intervalo de confianza para la diferencia de medias
    diff_means = media_grupo1 - media_grupo2
    pooled_std = np.sqrt(((n1-1)*std_grupo1**2 + (n2-1)*std_grupo2**2) / df)
    se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = diff_means - t_critical * se_diff
    ci_upper = diff_means + t_critical * se_diff
    
    # Interpretación contextual
    if es_significativo:
        if media_grupo1 > media_grupo2:
            interpretacion = f" Existe una diferencia estadísticamente significativa (p={p_value:.4f}). {nombre_grupo1} presenta valores significativamente MAYORES que {nombre_grupo2}."
        else:
            interpretacion = f" Existe una diferencia estadísticamente significativa (p={p_value:.4f}). {nombre_grupo2} presenta valores significativamente MAYORES que {nombre_grupo1}."
    else:
        interpretacion = f" No hay diferencia estadísticamente significativa entre {nombre_grupo1} y {nombre_grupo2} (p={p_value:.4f})."
    
    return {
        'suficientes_datos': True,
        'tipo_analisis': 'completo',
        't_statistic': t_statistic,
        'p_value': p_value,
        'grados_libertad': df,
        'es_significativo': es_significativo,
        'alpha': alpha,
        'media_grupo1': media_grupo1,
        'media_grupo2': media_grupo2,
        'std_grupo1': std_grupo1,
        'std_grupo2': std_grupo2,
        'n_grupo1': n1,
        'n_grupo2': n2,
        'nombre_grupo1': nombre_grupo1,
        'nombre_grupo2': nombre_grupo2,
        'diferencia_medias': diff_means,
        'intervalo_confianza': (ci_lower, ci_upper),
        'interpretacion': interpretacion
       }
        



# FUNCIÓN PARA AGREGAR AL PDF
def agregar_test_t_al_pdf(pdf, test_t_results, temp_files):
    
    
    if not test_t_results or not test_t_results.get('suficientes_datos', False):
        return
    
    pdf.add_page()
    
    # Título
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 10, "ANALISIS ESTADISTICO: TEST T DE STUDENT", 0, 1, 'C')
    pdf.ln(5)
    
    # Introducción
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, 
        "El Test T de Student se utilizo para determinar si existe una diferencia "
        "estadisticamente significativa entre los grupos analizados. "
        "Este analisis permite validar objetivamente la efectividad del tratamiento "
        f"comparando '{test_t_results['nombre_grupo1']}' vs '{test_t_results['nombre_grupo2']}'.",
        0, 'J')
    pdf.ln(5)
    
    # Cuadros de resultados principales
    y_start = pdf.get_y()
    
    # Estadístico t
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(102, 126, 234)
    pdf.rect(20, y_start, 55, 25, 'D')
    pdf.set_xy(23, y_start + 3)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Estadistico t", 0, 1)
    pdf.set_xy(23, y_start + 11)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 6, f"{test_t_results['t_statistic']:.4f}", 0, 1)
    pdf.set_xy(23, y_start + 19)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 4, f"gl = {test_t_results['grados_libertad']}", 0, 1)
    
    # Valor p
    pdf.rect(80, y_start, 55, 25, 'D')
    pdf.set_xy(83, y_start + 3)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Valor p", 0, 1)
    pdf.set_xy(83, y_start + 11)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 6, f"{test_t_results['p_value']:.4f}", 0, 1)
    
    # Significancia
    pdf.rect(140, y_start, 55, 25, 'D')
    pdf.set_xy(143, y_start + 3)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Significativo?", 0, 1)
    pdf.set_xy(143, y_start + 11)
    pdf.set_font('Helvetica', 'B', 16)
    if test_t_results['es_significativo']:
        pdf.set_text_color(76, 175, 80)
        pdf.cell(0, 6, "SI", 0, 1, 'C')
    else:
        pdf.set_text_color(244, 67, 54)
        pdf.cell(0, 6, "NO", 0, 1, 'C')
    
    pdf.set_y(y_start + 30)
    pdf.ln(5)
    
    # Tabla de estadísticas descriptivas
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(102, 126, 234)
    pdf.set_fill_color(248, 249, 250)
    pdf.cell(0, 8, "Estadisticas descriptivas por grupo", 0, 1, 'L', True)
    pdf.ln(2)
    
    # Encabezados de tabla
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(50, 7, "Grupo", 1, 0, 'C', True)
    pdf.cell(30, 7, "n", 1, 0, 'C', True)
    pdf.cell(45, 7, "Media", 1, 0, 'C', True)
    pdf.cell(45, 7, "Desv. Std", 1, 1, 'C', True)
    
    pdf.set_font('Helvetica', '', 9)
    
    # Fila Grupo 1
    pdf.cell(50, 7, test_t_results['nombre_grupo1'], 1, 0, 'C')
    pdf.cell(30, 7, str(test_t_results['n_grupo1']), 1, 0, 'C')
    pdf.cell(45, 7, f"{test_t_results['media_grupo1']:.2f}", 1, 0, 'C')
    pdf.cell(45, 7, f"{test_t_results['std_grupo1']:.2f}", 1, 1, 'C')
    
    # Fila Grupo 2
    pdf.cell(50, 7, test_t_results['nombre_grupo2'], 1, 0, 'C')
    pdf.cell(30, 7, str(test_t_results['n_grupo2']), 1, 0, 'C')
    pdf.cell(45, 7, f"{test_t_results['media_grupo2']:.2f}", 1, 0, 'C')
    pdf.cell(45, 7, f"{test_t_results['std_grupo2']:.2f}", 1, 1, 'C')
    
    # Fila Diferencia
    pdf.set_fill_color(255, 250, 205)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(50, 7, "Diferencia", 1, 0, 'C', True)
    pdf.cell(30, 7, "-", 1, 0, 'C', True)
    pdf.cell(45, 7, f"{test_t_results['diferencia_medias']:.2f}", 1, 0, 'C', True)
    pdf.cell(45, 7, "-", 1, 1, 'C', True)
    
    pdf.ln(5)
    
    # Crear gráfica
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('white')
    
    grupos = [test_t_results['nombre_grupo1'], test_t_results['nombre_grupo2']]
    medias = [test_t_results['media_grupo1'], test_t_results['media_grupo2']]
    errores = [test_t_results['std_grupo1'], test_t_results['std_grupo2']]
    
    bars = ax.bar(grupos, medias, yerr=errores, capsize=12, 
                 color=['#667eea', '#4facfe'], 
                 edgecolor='black', linewidth=2, alpha=0.85)
    
    for bar, media in zip(bars, medias):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{media:.2f}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    if test_t_results['es_significativo']:
        y_max = max([m + e for m, e in zip(medias, errores)])
        y_line = y_max * 1.15
        ax.plot([0, 1], [y_line, y_line], 'k-', linewidth=2)
        ax.plot([0, 0], [medias[0] + errores[0], y_line], 'k-', linewidth=2)
        ax.plot([1, 1], [medias[1] + errores[1], y_line], 'k-', linewidth=2)
        ax.text(0.5, y_line, f'p = {test_t_results["p_value"]:.4f} *', 
               ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Media ± Desviacion Estandar', fontsize=11, fontweight='bold')
    ax.set_title(f'Comparacion: {test_t_results["nombre_grupo1"]} vs {test_t_results["nombre_grupo2"]}', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Guardar gráfica
    tmp_graph = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(tmp_graph.name, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    temp_files.append(tmp_graph.name)
    
    # Insertar gráfica
    pdf.image(tmp_graph.name, x=25, w=160)
    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Grafica: Comparacion estadistica con barras de error (* p < 0.05)", 0, 1, 'C')
    pdf.ln(5)
    
    # Cuadro de interpretación
    pdf.set_fill_color(245, 250, 255)
    pdf.set_draw_color(102, 126, 234)
    pdf.rect(15, pdf.get_y(), 180, 5, 'DF')
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(102, 126, 234)
    pdf.ln(2)
    pdf.cell(0, 6, "Interpretacion estadistica", 0, 1, 'C')
    pdf.ln(3)
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(0, 0, 0)
    interpretacion_limpia = test_t_results['interpretacion'].replace('≥', '>=')
    interpretacion_limpia = ''.join(c for c in interpretacion_limpia if ord(c) < 128 or c == '\n')
    pdf.multi_cell(0, 5, interpretacion_limpia, 0, 'J')
    
    ci_lower, ci_upper = test_t_results['intervalo_confianza']
    pdf.ln(3)
    pdf.multi_cell(0, 5, 
        f"Intervalo de confianza 95% para la diferencia de medias: [{ci_lower:.2f}, {ci_upper:.2f}]",
        0, 'J')
    
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        "Nota: El intervalo de confianza indica el rango probable de la verdadera diferencia "
        "entre las medias poblacionales con un 95% de confianza. "
        "Si el intervalo NO contiene el cero, la diferencia es estadisticamente significativa.",
        0, 'J')

# DESCRIPCIÓN NARRATIVA DEL ENSAYO 
def generar_descripcion(norma, microorg_selec, medio, tiempo, temperatura, escala, dilucion=None):
    norma_lower = str(norma or "").lower()
    microorg_selec = str(microorg_selec or "").strip()
    micro = microorg_selec.lower()

    def formato_microorganismo_html(nombre):
        if not nombre:
            return "microorganismo no especificado"
        return f"<i>{nombre}</i>"
    
    descripcion_micro = ""
    descripcion_generada = ""

    # Descripción base por microorganismo 
    if "staphylococcus aureus" in micro:
        descripcion_micro = ("<i>Staphylococcus aureus</i> es una bacteria Gram positiva, "
                             "frecuentemente implicada en infecciones cutáneas y sistémicas. "
                             "Posee una elevada resistencia a condiciones adversas y es "
                             "empleada como microorganismo de referencia en ensayos de eficacia antimicrobiana.")
    elif "klebsiella pneumoniae" in micro:
        descripcion_micro = ("<i>Klebsiella pneumoniae</i> es una bacteria Gram negativa encapsulada, "
                             "asociada a infecciones respiratorias y hospitalarias. Su alta resistencia "
                             "a agentes antimicrobianos la convierte en un modelo útil para evaluar "
                             "la efectividad de superficies y materiales tratados.")
    elif "aspergillus niger" in micro:
        descripcion_micro = ("<i>Aspergillus niger</i> es un hongo filamentoso de crecimiento rápido, "
                             "frecuente en ambientes húmedos y suelos. Se utiliza como cepa de ensayo "
                             "en estudios de biodeterioro para determinar la resistencia fúngica de materiales.")
    elif "trichoderma virens" in micro:
        descripcion_micro = ("<i>Trichoderma virens</i> es un hongo con capacidad antagonista frente a otras especies fúngicas. "
                             "Se emplea en ensayos de biodeterioro debido a su habilidad para colonizar materiales orgánicos "
                             "y evaluar la durabilidad de superficies expuestas.")
    elif "escherichia coli" in micro:
        descripcion_micro = ("<i>Escherichia coli</i> es una bacteria Gram negativa, "
                             "usada como organismo indicador en pruebas de actividad antimicrobiana. "
                             "Su rápido crecimiento y comportamiento bien caracterizado la hacen ideal "
                             "para determinar la eficacia de materiales bactericidas.")
    elif "streptomyces" in micro:
        descripcion_micro = ("<i>Streptomyces species</i> pertenece a las actinobacterias, "
                             "caracterizadas por su crecimiento filamentoso y su capacidad de producir antibióticos. "
                             "Se usa en la norma ASTM E1428 para evaluar el biodeterioro y resistencia de materiales.")
    else:
        descripcion_micro = ("Microorganismo de ensayo empleado para determinar la respuesta del material "
                             "frente a condiciones microbiológicas controladas.")

    microorg_html = f"<i>{microorg_selec}</i>" 
    
    # Descripción por norma 
    if "aatcc" in norma_lower or "tm147" in norma_lower:
        descripcion_general = (f"El presente ensayo se realizó conforme a la norma {norma}, "
                               f"empleando como microorganismo de prueba a <i>{microorg_selec}</i>. "
                               f"La muestra se cultivó en {medio if medio else 'medio no especificado'} durante "
                               f"{tiempo if tiempo else 'tiempo no definido'} horas a una temperatura de "
                               f"{temperatura} °C. "
                               f"Este procedimiento tiene como objetivo determinar la actividad antibacteriana "
                               "por difusión en superficie de contacto.")
        
    elif "astm g21" in norma_lower or "g21" in norma_lower:
        descripcion_general = (f"El ensayo se llevó a cabo bajo los lineamientos de la norma {norma}, "
                               f"empleando a <i>{microorg_selec}</i> como organismo de ensayo para evaluar la resistencia fúngica del material. "
                               f"Las muestras fueron incubadas en {medio if medio else 'medio de cultivo adecuado'}"
                               f"durante {tiempo if tiempo else 'tiempo definido por el método'} a una temperatura de "
                               f"{temperatura} °C. "
                               "y clasificar el grado de biodeterioro conforme a la escala ASTM G21-15.")
        
    elif "jis" in norma_lower or "z2801" in norma_lower:
        descripcion_general = (f"El análisis se realizó siguiendo la norma {norma}, "
                               f"utilizando a <i>{microorg_selec}</i> como microorganismo representativo para la evaluación "
                               "de la eficacia antimicrobiana de superficies. "
                               f"Las muestras fueron incubadas en {medio if medio else 'medio adecuado'} "
                               f"durante {tiempo if tiempo else 'tiempo definido'} horas a {temperatura} °C, "
                               f"empleando una dilución de {dilucion if dilucion else 'no especificada'}. "
                               f"permitiendo calcular la reducción logarítmica (R) como indicador cuantitativo "
                               "de la eficacia del tratamiento aplicado.")
        
    elif "astm e1428" in norma_lower or "e1428" in norma_lower:
        descripcion_general = (f"El presente análisis se efectuó de acuerdo con la norma {norma}, "
                               f"empleando a <i>{microorg_selec}</i> para evaluar la resistencia al biodeterioro de materiales expuestos "
                               "a actinomicetos. "
                               f"Las muestras fueron cultivadas en {medio if medio else 'medio adecuado'}, "
                               f"durante {tiempo if tiempo else 'tiempo definido'} horas a una temperatura de {temperatura} °C. "
                               f"permitió identificar el grado de crecimiento y estimar la capacidad del material "
                               "para resistir la colonización microbiana.")
    else:
        descripcion_general = (f"El ensayo se desarrolló bajo condiciones controladas con {microorg_selec}, "
                               "siguiendo protocolos microbiológicos estándar para la evaluación de materiales.")

    # Devolver texto justificado en HTML
    st.markdown(descripcion_generada, unsafe_allow_html=True)
    descripcion_completa = f"{descripcion_general}<br><br>{descripcion_micro}"
    return f"<div style='text-align: justify;'>{descripcion_completa}</div>"

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def mostrar_resultado_individual(replica, norma, analyzer, mm_per_pixel):
    orig = replica['original']
    pca = replica['pca']
    ms = replica['meanshift']
    results = replica['results']

    st.markdown("#### Imágenes procesadas")
    cols = st.columns(2)
    cols[1].image(orig, caption="Original", use_container_width=True)
    #cols[1].image(pca, caption="PCA", use_container_width=True)
    #cols[2].image(ms, caption="MeanShift", use_container_width=True)

    # Imagen final según norma
    processed_img = None
    # --- AATCC TM147 ---
    if 'AATCC' in norma or 'TM147' in norma:
        # ⭐ CORRECCIÓN: Capturar 6 valores
        mask_textil, mask_microbio, avg_halo, overlay_img, measurements, halo_center = \
            analyzer.analyze_halo_TM147_visual_final(orig, mm_per_pixel, debug=False)

        processed_img = overlay_img.copy()

        # Guardar información
        cx_textil, cy_textil, r_textil = halo_center
        st.session_state['halo_center'] = (cx_textil, cy_textil)
        st.session_state['halo_specimen_radius'] = r_textil

        # Mostrar resultados
        if measurements and len(measurements) > 0:
            st.success(f"✅ Halo detectado: {avg_halo:.2f} mm ({len(measurements)} mediciones)")
    
    elif 'G21' in norma or 'ASTM_G21' in norma:
        # Analizar crecimiento fúngico
        rating, coverage_percentage, fungal_mask = analyzer.analyze_fungal_growth(
            orig_img, ms, microorganism=microorg_selec  # ← Pasa el microorganismo
        )
        
        if fungal_mask is not None and np.sum(fungal_mask > 0) > 0:
            overlay = orig_img.copy()
            
            # ===== DETERMINAR COLOR SEGÚN MICROORGANISMO =====
            micro_lower = microorg_selec.lower()
            
            if 'aspergillus' in micro_lower or 'niger' in micro_lower:
                # ASPERGILLUS (negro) → Overlay ROJO OSCURO
                color = [0, 180, 0]  # RGB: verde más visible
                print(f"🟢 Aplicando overlay VERDE para Aspergillus")
                
            elif 'trichoderma' in micro_lower or 'virens' in micro_lower or 'viride' in micro_lower:
                # TRICHODERMA (blanco) → Overlay VERDE BRILLANTE
                color = [0, 180, 0]  # RGB: verde más visible
                print(f"🟢 Aplicando overlay VERDE para Trichoderma")
                
            else:
                # GENÉRICO → Verde por defecto
                color = [0, 255, 0]
                print(f"⚪ Aplicando overlay VERDE genérico")
            
            # Aplicar color donde detectó hongo
            overlay[fungal_mask > 0] = color
            
            # Mezclar con imagen original (60% original, 40% overlay para mayor visibilidad)
            processed_img = cv2.addWeighted(orig_img, 0.4, overlay, 0.6, 0)
            
            print(f"✓ Overlay aplicado: {np.sum(fungal_mask > 0)} píxeles coloreados")
        else:
            processed_img = orig_img.copy()
            print("⚠ No se detectó crecimiento fúngico")


    elif 'G21' in norma or 'ASTM_G21' in norma:
        print(f"\n{'='*60}")
        print(f"🔬 PROCESANDO IMAGEN PARA ASTM G21")
        print(f"   Microorganismo: {microorg_selec}")
        print(f"{'='*60}\n")
        
        rating, coverage_percentage, fungal_mask = analyzer.analyze_fungal_growth(
            orig, ms, microorganism=microorg_selec
        )
        
        if fungal_mask is not None and np.sum(fungal_mask > 0) > 0:
            overlay = orig.copy()
            
            # Determinar color según microorganismo
            micro_lower = microorg_selec.lower()
            
            if 'aspergillus' in micro_lower or 'niger' in micro_lower:
                color = [180, 50, 50]  # Rojo oscuro
                print("🔴 Aplicando overlay ROJO para Aspergillus")
            elif 'trichoderma' in micro_lower:
                color = [0, 255, 100]  # Verde brillante
                print("🟢 Aplicando overlay VERDE para Trichoderma")
            else:
                color = [0, 255, 0]
                print("⚪ Aplicando overlay VERDE genérico")
            
            # Crear overlay con 50% de transparencia para mejor visibilidad
            overlay[fungal_mask > 0] = color
            processed_img = cv2.addWeighted(orig, 0.5, overlay, 0.5, 0)
            
            print(f"✅ Overlay aplicado: {np.sum(fungal_mask > 0)} píxeles coloreados")
            print(f"   Cobertura: {coverage_percentage:.2f}%")
            print(f"   Rating: {rating}/4\n")
        else:
            processed_img = orig_img.copy()
            print("⚠️ No se detectó crecimiento fúngico\n")
            

    elif 'JIS' in norma or 'Z2801' in norma:
        # 🆕 Usar función optimizada para JIS
        # Contar y obtener colonias válidas
        treated_count, treated_colonies = analyzer.count_colonies_opencv(orig, ms)

        # Copiar imagen original
        processed_img = orig.copy()

        # Crear una máscara vacía para todas las colonias
        mask_colonies = np.zeros(orig.shape[:2], dtype=np.uint8)

        # Pintar cada colonia en la máscara
        for i, colony in enumerate(treated_colonies, 1):
            cv2.drawContours(mask_colonies, [colony], -1, i, -1)  # Cada colonia tiene un valor único

        # Crear imagen en color para superposición
        overlay = np.zeros_like(processed_img)

        # Asignar un color a cada colonia
        for i in range(1, len(treated_colonies)+1):
            overlay[mask_colonies == i] = (0, 255, 0)  # Verde, puedes cambiar por otro color

        # Mezclar overlay con la original usando alpha blending
        alpha = 0.5
        processed_img = cv2.addWeighted(processed_img, 1 - alpha, overlay, alpha, 0)

        # Agregar contador
        cv2.putText(
            processed_img,
            f"Colonias: {treated_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        # Mostrar en Streamlit
        cols[3].image(processed_img, caption="Resultado final", use_container_width=True)

def plot_results_by_norm(norma, results):
    norma_lower = str(norma).lower()

    # colores (solo dos)
    col1 ="#667eea"
    col2 ="#4facfe"
 
    # JIS (comparativa control vs tratada)
    if "jis" in norma_lower or "z2801" in norma_lower:
        control = results.get('control_count') or results.get('control') or results.get('control_count', None)
        tratada = results.get('treated_count') or results.get('tratada') or results.get('treated_count', None)
        log_red = results.get('log_reduction') or results.get('reduccion_log') or results.get('reduccion_log', None)

        if control is not None and tratada is not None:
            fig, ax = plt.subplots(figsize=(8,4))
            labels = ["Control", "Tratada"]
            values = [control, tratada]
            bars = ax.bar(labels, values, color=[col1, col2], edgecolor='black', linewidth=1)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{bar.get_height():.0f}', ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel("Recuento (UFC / conteo)", fontsize=11)
            ax.set_title("Control vs Tratada (JIS Z 2801)", fontsize=12)
            ax.grid(axis='y', alpha=0.25, linestyle='--')
             #  Ajustar límites del eje Y para incluir 0
            ax.set_ylim(bottom=0)

            st.pyplot(fig)
            plt.close(fig)

            if isinstance(log_red, (int, float)):
                # ADVERTENCIA si es negativo
                if log_red < 0:
                    st.error(f" **Reducción logarítmica negativa:** {log_red:.2f}")
                    st.warning("Esto indica que el tratamiento **aumentó** el crecimiento bacteriano en lugar de reducirlo.")
                else:
                    st.info(f"**Reducción logarítmica (R):** {log_red:.2f}")
        else:
            st.warning("Datos incompletos para JIS: falta 'control' o 'tratada' en results.")

    #  AATCC TM147: halo de inhibición 
    elif "aatcc" in norma_lower or "tm147" in norma_lower:
        
        # Buscar el halo con diferentes nombres posibles
        halo = results.get('inhibition_halo_mm')

        if halo is not None and isinstance(halo, (int, float)):
            #  Forzar a 0 si es negativo
            halo = max(0, halo)

            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(["Halo (mm)"], [halo], color=[col1], edgecolor='black')
            ax.set_ylim(bottom=0, top=max(halo*1.3, 1))  #  Siempre desde 0
            ax.set_ylabel("mm", fontsize=11)
            ax.set_title("Halo de inhibición (AATCC TM147)", fontsize=12)
            ax.text(0, halo, f"{halo:.2f} mm", ha='center', va='bottom', fontweight='bold')
            ax.grid(axis='y', alpha=0.2, linestyle='--')
            
        
            st.pyplot(fig)
            plt.close(fig)

    #  ASTM G21: cobertura fúngica y rating 
    elif "g21" in norma_lower or "astm g21" in norma_lower:
        coverage = results.get('coverage_percentage') or results.get('cobertura') 
        rating = results.get('astm_g21_rating') or results.get('rating')
        if coverage is not None and rating is not None:
            coverage = max(0, min(100, coverage))  #  Limitar 0-100
            rating_pct = (rating / 4.0) * 100
            
            fig, ax = plt.subplots(figsize=(8,4))
            labels = ["Cobertura (%)", "Rating (0-4) -> %"]
            values = [coverage, rating_pct]
            bars = ax.bar(labels, values, color=[col1, col2], edgecolor='black')
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val, 
                       f"{val:.2f}", ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylim(bottom=0, top=max(values)*1.15)  #  Desde 0
            ax.set_title("Cobertura fúngica y rating (ASTM G21)", fontsize=12)
            ax.grid(axis='y', alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Rating ASTM G21: {rating} / 4 (mostrado como % para comparar)")
        elif coverage is not None:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(["Cobertura (%)"], [coverage], color=[col1], edgecolor='black')
            ax.set_ylim(0, max(coverage*1.3, 1))
            ax.set_title("Cobertura fúngica (ASTM G21)", fontsize=12)
            ax.text(0, coverage, f"{coverage:.2f} %", ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No se encontraron 'coverage_percentage' ni 'astm_g21_rating' en results para ASTM G21.")

    # ASTM E1428: colonias / porcentaje de área 
    elif "astm e1428" in norma_lower or "e1428" in norma_lower:
        colony_count = results.get('colony_count') or results.get('colonias') or None
        coverage = results.get('coverage_percentage') or results.get('coverage') or None

        if coverage is not None:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(["Área colonizada (%)"], [coverage], color=[col1], edgecolor='black')
            ax.set_ylim(0, max(coverage*1.3, 1))
            ax.set_title("Porcentaje de área colonizada (ASTM E1428)", fontsize=12)
            ax.text(0, coverage, f"{coverage:.2f} %", ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)
        elif colony_count is not None:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(["Colonias detectadas"], [colony_count], color=[col1], edgecolor='black')
            ax.set_title("Conteo de colonias (ASTM E1428)", fontsize=12)
            ax.text(0, colony_count, f"{int(colony_count)}", ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No se encontraron 'coverage_percentage' ni 'colony_count' en results para ASTM E1428.")

    #  Caso por defecto: graficar numéricos disponibles (solo tratada)
    else:
        numeric_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        if numeric_results:
            fig, ax = plt.subplots(figsize=(8,4))
            keys = list(numeric_results.keys())
            values = list(numeric_results.values())
            # usar solo dos colores alternando
            colors = [col1 if i % 2 == 0 else col2 for i in range(len(keys))]
            bars = ax.bar(keys, values, color=colors, edgecolor='black')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2f}", ha='center', va='bottom', fontweight='bold')
            ax.set_title("Resultados cuantitativos (tratada)", fontsize=12)
            ax.grid(axis='y', alpha=0.2)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No hay resultados numéricos para graficar (solo datos cualitativos).")

#  Inicializar página 
if "pagina" not in st.session_state:
    st.session_state["pagina"] = "inicio"

# DEBUG: Mostrar estado actual (comentar después de arreglar)
st.sidebar.write(f"**DEBUG - Página actual:** {st.session_state['pagina']}")

#  PÁGINA DE INICIO 
if st.session_state["pagina"] == "inicio":

    add_bg_from_local("imagen_inicio.png")

    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    st.markdown("""
                <style>
                div.stButton > button:first-child {
        display: block;
        margin-left: 160%;
        font-size: 32px !important;
        padding: 20px 60px !important;
        border-radius: 15px !important;
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("INICIO"):
      st.session_state["pagina"] = "parametros"

     # Título y subtítulo en color negro, centrados
    #  Texto debajo del botón 
    st.markdown("""
        <div style='text-align:center; color:black; margin-top:120px;'>
            <h1 style='font-size:20px; font-weight:600;'> Bioingeniería</h1>
            <h2 style='font-size:18px; font-weight:400;'>Pontificia Universidad Javeriana</h2>
        </div>
    """, unsafe_allow_html=True)

#  PÁGINA DE PARÁMETROS 
elif st.session_state["pagina"] == "parametros":

    add_bg_from_local("Fondo_parametros.png")

     # DESCRIPCIÓN DEL APLICATIVO 
    st.markdown("""
    <div style='text-align: justify;'>
    <h3> Descripción del aplicativo</h3>

    El aplicativo <b>MicroVision</b> fue desarrollado como una herramienta computacional para el análisis de imágenes microbiológicas, con el propósito de evaluar la actividad antimicrobiana, según los estándares internacionales AATCC TM147, ASTM G21, JIS Z 2801 y ASTM E1428.

    Mediante técnicas de procesamiento digital de imágenes, el sistema identifica y cuantifica parámetros como el <i>halo de inhibición</i> y el <i>conteo de colonias</i>, dependiendo del tipo de ensayo. 

    </br>
    <p> A continuación, diligencie la información requerida para iniciar el análisis, incluyendo la norma de referencia, el microorganismo de prueba, las condiciones de cultivo y las imágenes obtenidas durante el ensayo.</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("Selección de parámetros experimentales")

    normas = ["AATCC TM147-2011(2016)", "ASTM G21-15", "JIS Z 2801 2010", "ASTM E1428-24"]
    norma = st.selectbox("Selecciona la norma:", normas)

    micro_list = [
        "Staphylococcus aureus",
        "Klebsiella pneumoniae",
        "Aspergillus niger",
        "Trichoderma virens",
        "Escherichia coli",
        "Streptomyces species",
    ]
    microorg_selec = st.selectbox("Selecciona el microorganismo:", micro_list)

    # Carpeta base, compatible con Python normal y ejecutable (.exe)
    if getattr(sys, 'frozen', False):
        # Si está dentro del ejecutable compilado
        base_path = sys._MEIPASS
    else:
        # Si se ejecuta como script normal
        base_path = os.path.dirname(os.path.abspath(__file__))

    # Carpeta donde están las imágenes (dentro del proyecto o del exe)
    carpeta_imagenes = os.path.join(base_path, "Imagenes_tesis_microorganismos")

    imagenes = {
        "escherichia coli": os.path.join(carpeta_imagenes, "Escherichia_coli.PNG"),
        "klebsiella pneumoniae": os.path.join(carpeta_imagenes, "Klebsiella_pneumoniae.PNG"),
        "staphylococcus aureus": os.path.join(carpeta_imagenes, "Staphylococcus_aureus.PNG"),
        "aspergillus niger": os.path.join(carpeta_imagenes, "Aspergillus_niger.PNG"),
        "trichoderma virens": os.path.join(carpeta_imagenes, "Trichoderma_virens.png"),
        "streptomyces species": os.path.join(carpeta_imagenes, "Streptomyces_species.PNG")
    }

    microorg_norm = microorg_selec.strip().lower()
    texto = "Información no registrada"
    refText = "Referencia no disponible."

    if "escherichia coli" in microorg_norm:
        texto = """• Tipo: Bacteria Gram negativa  
            • Indicador de contaminación fecal  
            • Hábitat: Intestinos de humanos y animales  
            • Patogenicidad: Algunas cepas son patógenas (ej. E. coli O157:H7)  
            • Crecimiento óptimo: 37 °C  
            • Morfología colonial: Colonias circulares, lisas, convexas, de borde entero, color blanco-grisáceo."""
        refText = "Imágenes tomadas de: Tinción de Gram en Escherichia coli: https://es.wikipedia.org/wiki/Archivo:Bacteria_gram_negativa,_Escherichia_coli.jpg; morfología de Escherichia coli: https://microbialnotes.com/isolation-and-identification-of-escherichia-coli-e-coli#google_vignette; SEM de Escherichia coli: https://www.researchgate.net/figure/SEM-image-for-Escherichia-Coli_fig3_313029996, TEM de Escherichia coli: https://www.sciencephoto.com/media/864718/view/e-coli-bacterium-tem"
       
    elif "klebsiella pneumoniae" in microorg_norm:
        texto = """• Tipo: Bacteria Gram negativa  
            • Patógeno oportunista asociado a infecciones nosocomiales  
            • Posee cápsula que le confiere resistencia a la fagocitosis  
            • Hábitat: Tracto respiratorio y digestivo  
            • Crecimiento óptimo: 35–37 °C  
            • Morfología colonial: Colonias grandes, mucoides, brillantes, de borde regular."""
        refText = "Imágenes tomadas de: Morfologia de Klebsiella pneumoniae: https://www.microbiologyinpictures.com/bacteria-photos/klebsiella-pneumoniae-photos/klebsiella-pneumoniae-colonies-appearance.html, tinción de Gram en Klebsiella pneumoniae:  https://www.atsu.edu/faculty/chamberlain/website/kpneumo.htm; SEM de Klebsiella pneumoniae: https://www.sciencephoto.com/media/12329/view/sem-of-klebsiella-pneumoniae-bacteria, TEM de Klebsiella pneumoniae: https://www.sciencephoto.com/media/1148716/view/klebsiella-pneumoniae-bacteria-tem"
      
    elif "staphylococcus aureus" in microorg_norm:
        texto = """• Tipo: Bacteria Gram positiva  
            • Productora de toxinas y enzimas  
            • Hábitat: Piel, mucosas y ambiente hospitalario  
            • Patogenicidad: Infecciones cutáneas y sistémicas (ej. septicemia)  
            • Crecimiento óptimo: 35–37 °C  
            • Morfología colonial: Colonias circulares, convexas, lisas, color dorado o amarillento."""
        refText = "Imágenes tomadas de:  Tinción de Gram de Staphylococcus aureus:   https://www.researchgate.net/figure/Staphylococcus-aureus-under-microscope-Adapted-from-Foster-2017_fig3_355265910, Morfología de Staphylococcus aureus: https://www.google.com/imgres?imgurl=https%3A%2F%2Fmicrobenotes.com%2Fwp-content%2Fuploads%2F2017%2F11%2FStaphylococcus-aureus-on-Tryptic-Soy-Agar.jpg&tbnid=0kacTe9qfh_efM&vet=1&imgrefurl=https%3A%2F%2Fmicrobenotes.com%2Fstaphylococcus-aureus%2F&docid=dkJWLItbBVevYM&w=800&h=640&source=sh%2Fx%2Fim%2Fm1%2F1&kgs=0069edf512f29ce7&shem=isst, TEM de Staphylococcus aureus: https://www.sciencephoto.com/media/458315/view/staphylococcus-aureus-tem, SEM de Staphylococcus aureus: https://www.sciencephoto.com/media/12898/view/sem-of-staphylococcus-aureus-bacteria"
      
    elif "aspergillus niger" in microorg_norm:
        texto = """• Tipo: Hongo filamentoso  
            • Se utiliza como organismo de prueba en ensayos de biodeterioro  
            • Hábitat: Suelo, materia orgánica en descomposición  
            • Relevancia: Puede crecer en textiles y producir manchas negras  
            • Crecimiento óptimo: 25–30 °C  
            • Morfología colonial: Colonias algodonosas, de color inicialmente blanco que se tornan negras con la esporulación."""
        refText = "Imágenes tomadas de: Tinción de Gram en Aspergillus niger: https://www.heraldo.es/noticias/aragon/2020/08/31/que-es-el-hongo-aspergillus-1393297.html, Morfologia de Aspergillus niger: http://svdb.minec.gob.ve/flora/aspergillus-niger, SEM de  Aspergillus niger: https://www.researchgate.net/figure/Morphological-characteristics-of-Aspergillus-niger-AF-using-light-microscope-and_fig5_281271289"
      
    elif "trichoderma virens" in microorg_norm:
        texto = """• Tipo: Hongo filamentoso  
            • Conocido por su capacidad de producir metabolitos antifúngicos  
            • Hábitat: Suelo y ambientes ricos en materia orgánica  
            • Relevancia: Se emplea en ensayos de biodeterioro de materiales  
            • Crecimiento óptimo: 25–30 °C  
            • Morfología colonial: Colonias algodonosas, inicialmente blancas, que se tornan verde-azuladas por la esporulación."""
        refText = "Imágenes tomadas de: Tinción de Gram en Trichoderma Virens: https://www.google.com/url?sa=i&url=https%3A%2F%2Falchetron.com%2FGliocladium&psig=AOvVaw3yY3ssmHfeTKxcZCkdlzci&ust=1756996785507000&source=images&cd=vfe&opi=89978449&ved=0CBgQjhxqFwoTCOCQ-6LpvI8DFQAAAAAdAAAAABAE, SEM de Trichoderma Virens: https://www.google.com/url?sa=i&url=https%3A%2F%2Fdialnet.unirioja.es%2Fdescarga%2Farticulo%2F5644956.pdf&psig=AOvVaw0d3gWOJqAER7vzSwClY8FI&ust=1756996854397000&source=images&cd=vfe&opi=89978449&ved=0CBgQjhxqGAoTCJDT4r3pvI8DFQAAAAAdAAAAABCuAQ , Morfologia de Trichoderma Virens: https://biologicalslatam.com/trichoderma-el-hongo-que-fortalece-cultivos-y-combate-plagas/"
      
    elif "streptomyces" in microorg_norm:
        texto = """• Tipo: Bacteria filamentosa (Actinobacteria)  
            • Productora natural de antibióticos  
            • Hábitat: Suelo (olor característico a tierra mojada por geosmina)  
            • Relevancia: Usada en ASTM E1428-24 para evaluar biodeterioro  
            • Crecimiento óptimo: 25–30 °C  
            • Morfología colonial: Colonias secas, pulverulentas, aspecto aterciopelado, generalmente blancas o grises."""
        refText = "Imágenes tomadas de: Tinción de Gram en Streptomyces species: Elaborado por Carolina Santos Baquero, Morfología de Streptomyces species: https://bacdive.dsmz.de/strain/15058, SEM de Streptomyces species: https://www.researchgate.net/figure/SEM-image-of-Streptomyces-sp-UKMCC-PT15_fig1_310898194"
      
    if st.button("Mostrar ficha técnica"):
        col_img, col_text = st.columns([1, 2])
        
        with col_img:
            if microorg_norm in imagenes and os.path.exists(imagenes[microorg_norm]):
                st.image(imagenes[microorg_norm], width=250)

                # Pie de figura: referencia en letra pequeña y gris
                st.markdown(
                    f"""
                    <p style='
                        font-size:8.5px;
                        color:#777;
                        text-align:justify;
                        font-style:italic;
                        line-height:1.05;
                        margin-top:5px;
                        text-justify:inter-word;
                    '>
                        {refText}
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("Imagen no disponible")
        
        with col_text:
            st.info(f"### Ficha técnica de *{microorg_selec}*\n\n{texto}")

    st.markdown("---")
    st.markdown("### Condiciones experimentales")
    medio = st.text_input("Medio de cultivo (ej: Agar nutritivo)")
    tiempo = st.text_input("Tiempo de incubación (h)")
    temperatura = st.text_input("Temperatura (°C)")
    dilucion = None
    if norma == "JIS Z 2801 2010":
        dilucion = st.text_input("Dilución utilizada (ej: 1:100)")
    
   # Calibración automática de escala
    st.markdown("###  Calibración de escala")
    
    col_calib1, col_calib2 = st.columns([2, 1])
    
    with col_calib1:
        metodo_calibracion = st.radio(
            "Método de calibración:",
            ["Automático", "Manual"],
            help="El método automático detecta la placa Petri y calcula la escala"
        )
    
    with col_calib2:
        if metodo_calibracion == "Automático":
            diametro_placa = st.number_input(
                "Diámetro de la caja de Petri (mm)", 
                min_value=50.0, 
                max_value=150.0, 
                value=90.0,
                step=5.0,
                help="Diámetro estándar: 90mm o 60mm"
            )
            escala = None  # Se calculará automáticamente
        else:
            diametro_placa = None
            escala = st.number_input(
                "Escala (mm por pixel)", 
                min_value=0.0, 
                format="%.5f", 
                value=0.05,
                help="Solo si conoces el valor exacto"
            )

    #  Guardar la información en el estado 
    st.session_state["norma"] = norma
    st.session_state["microorg_selec"] = microorg_selec
    st.session_state["medio"] = medio 
    st.session_state["tiempo"] = tiempo 
    st.session_state["temperatura"] = temperatura
    st.session_state["dilucion"] = dilucion
    st.session_state["escala"] = escala

        # CARGA DE IMÁGENES (AHORA CON RÉPLICAS) 
    imagenes_subidas = {}

    if norma == "JIS Z 2801 2010":
        st.markdown("###  Adjuntar imágenes (puedes subir réplicas)")

        imagenes_subidas["control"] = st.file_uploader(
            "Imágenes CONTROL (una o más)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )
        imagenes_subidas["tratada"] = st.file_uploader(
            "Imágenes TRATADAS (una o más)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )
    else:
        st.markdown("### Adjuntar imágenes tratadas (una o más)")
        imagenes_subidas["tratada"] = st.file_uploader(
            "Imágenes TRATADAS", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )

    # Guardar en session_state
    st.session_state["imagenes_subidas"] = imagenes_subidas

    if norma == "JIS Z 2801 2010":
        if imagenes_subidas.get("control"):
            st.info(" Imagen control adjuntada correctamente.")
        if imagenes_subidas.get("tratada"):
            st.info(" Imagen tratada adjuntada correctamente.")
    else:
        if imagenes_subidas.get("tratada"):
            st.info(" Imagen tratada adjuntada correctamente.")

    imagen_tratada = imagenes_subidas.get("tratada")
    imagen_control = imagenes_subidas.get("control")

    #st.session_state["imagen_tratada"] = imagen_tratada
    #st.session_state["imagen_control"] = imagen_control    

    # ANÁLISIS DE IMAGEN 
    st.markdown("---")
    # FUNCIÓN AUXILIAR PARA MOSTRAR IMÁGENES 
    def show_image(image, title=""):
        """Muestra una imagen en Streamlit con su título y verificación de existencia."""

        if image is not None:
            st.image(image, caption=title, use_container_width=True)
        else:
            st.warning(f"No se encontró imagen para '{title}'.")

        #  Crear analizador solo si no existe 
    if "analyzer" not in st.session_state:
        st.session_state["analyzer"] = MultiStandardAnalyzer()
    analyzer = st.session_state["analyzer"]
    
    if st.button(" Analizar imagen"):
       
        # Crear carpeta temporal
        tmp_dir = tempfile.mkdtemp()
        #st.write(f"Carpeta temporal creada en: {tmp_dir}")

         #  Definir rutas para imágenes subidas 
            #  INICIALIZAR TODAS LAS LISTAS AL INICIO
        treated_results_list = []
        control_results_list = []
        paths_tratadas = []
        paths_control = []

        #  GUARDAR IMÁGENES TRATADAS 
        if imagenes_subidas.get("tratada"):
            for idx, img in enumerate(imagenes_subidas["tratada"]):
                # Evitar espacios y caracteres especiales en el nombre
                safe_name = img.name.replace(" ", "_").replace("á", "a").replace("é", "e") \
                                    .replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
                temp_path = os.path.join(tmp_dir, f"tratada_{idx}_{safe_name}")

                # Guardar el archivo en disco
                try:
                    with open(temp_path, "wb") as f:
                        f.write(img.getbuffer())  # getbuffer() evita problemas de puntero de archivo
                    paths_tratadas.append(temp_path)
                    st.success(f"Imagen tratada guardada correctamente: {temp_path}")
                except Exception as e:
                    st.error(f"No se pudo guardar la imagen tratada: {safe_name}. Error: {e}")

        #  GUARDAR IMÁGENES DE CONTROL 
        if imagenes_subidas.get("control"):
            for idx, img in enumerate(imagenes_subidas["control"]):
                safe_name = img.name.replace(" ", "_").replace("á", "a").replace("é", "e") \
                                    .replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
                temp_path = os.path.join(tmp_dir, f"control_{idx}_{safe_name}")

                try:
                    with open(temp_path, "wb") as f:
                        f.write(img.getbuffer())
                    paths_control.append(temp_path)
                    st.success(f"Imagen de control guardada correctamente: {temp_path}")
                except Exception as e:
                    st.error(f"No se pudo guardar la imagen de control: {safe_name}. Error: {e}")

        # --- VERIFICAR QUE LAS IMÁGENES EXISTEN ANTES DE PROCESAR ---
        for path in paths_tratadas + paths_control:
            if not os.path.exists(path):
                st.error(f"La imagen no se encuentra en disco: {path}")
            #else:
              #  st.write(f"Imagen lista para procesar: {path}")

        # --- Calcular escala ---
        if metodo_calibracion == "Automático":
            if paths_tratadas:  # verificar que haya imágenes cargadas
                first_img = cv2.imread(paths_tratadas[0])
                if first_img is None:
                    st.error(f"❌ No se pudo cargar la imagen de referencia: {paths_tratadas[0]}")
                    mm_per_pixel = 0.05
                else:
                    # Convertir a RGB porque OpenCV carga en BGR
                    first_img_rgb = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)

                    # Calibrar la escala automáticamente
                    mm_per_pixel, _ = analyzer.calibrar_escala_automatica(first_img_rgb, diametro_placa)

                    # Validar el valor calculado
                    if mm_per_pixel > 0 and mm_per_pixel < 5:
                        st.success(f" Escala calibrada automáticamente: {mm_per_pixel:.5f} mm/pixel")
                    else:
                        st.warning(f" Valor de escala fuera de rango: {mm_per_pixel:.5f} mm/pixel — se usará 0.05 por defecto.")
                        mm_per_pixel = 0.05
            else:
                st.error(" No se subieron imágenes tratadas para calibrar automáticamente.")
                mm_per_pixel = 0.05  # valor por defecto
        else:
            mm_per_pixel = escala if escala else 0.05
            st.info(f" Usando escala manual: {mm_per_pixel:.5f} mm/pixel")


        #  PROCESAR CONTROLES (solo si existen) 
        if paths_control:
            st.markdown("##  Imágenes de CONTROL")
            for idx, control_path in enumerate(paths_control):
                ctrl_img_rgb, ctrl_pca, ctrl_ms = analyzer.load_and_process_image(control_path)
                ctrl_count, ctrl_colonies = analyzer.count_colonies_opencv(ctrl_img_rgb, ctrl_ms, debug=True)

                ctrl_processed = ctrl_img_rgb.copy()
                #cv2.drawContours(ctrl_processed, ctrl_colonies, -1, (0, 255, 0), 2)

                control_results_list.append({
                    'original': ctrl_img_rgb,
                    'pca': ctrl_pca,
                    'meanshift': ctrl_ms,
                    'processed': ctrl_processed,
                    'count': ctrl_count
                })

                st.markdown(f"### Control réplica {idx+1}")
                cols = st.columns(2)
                cols[0].image(ctrl_img_rgb, caption="Original", use_container_width=True)
                #cols[1].image(ctrl_pca, caption="PCA", use_container_width=True)
                #cols[2].image(ctrl_ms, caption="MeanShift", use_container_width=True)
                cols[1].image(ctrl_processed, caption="Colonias detectadas", use_container_width=True)
                st.metric("Colonias detectadas", ctrl_count)
                st.markdown("---")

            # Guardar primera réplica de control en session_state
            control_img_rgb = control_results_list[0]['original']
            control_processed = control_results_list[0]['processed']
            st.session_state["control_img_rgb"] = control_img_rgb
            st.session_state["control_processed"] = control_processed

        promedio_control = np.mean([c['count'] for c in control_results_list]) if control_results_list else None

        # ===== PROCESAR TRATADAS =====
        st.markdown("##  Imágenes TRATADAS")

        for idx, path_tratada in enumerate(paths_tratadas):
    
            if 'mm_per_pixel' not in locals() or mm_per_pixel is None:
                mm_per_pixel = 0.05

            # ⭐ CORRECCIÓN: Una sola llamada con variables correctas
            results, orig_img, pca, ms, _, _, _, _ = analyzer.analyze_by_standard(
                path_tratada, microorg_selec, norma, mm_per_pixel,
                control_count=promedio_control
            )

            # Crear imagen procesada según la norma
            processed_img = orig_img.copy()
            
            # === AATCC TM147 ===
            if 'AATCC' in norma or 'TM147' in norma:
                # ⭐ Volver a llamar para obtener la imagen con overlay visual
                mask_textil, mask_microbio, avg_halo, overlay_img, measurements, halo_center = \
                    analyzer.analyze_halo_TM147_visual_final(orig_img, mm_per_pixel, debug=False)
                processed_img = overlay_img.copy()
                
                # Guardar información adicional
                cx_textil, cy_textil, r_textil = halo_center
                st.session_state['halo_center'] = (cx_textil, cy_textil)
                st.session_state['halo_specimen_radius'] = r_textil

                # Mostrar resultados
                if measurements and len(measurements) > 0:
                    st.success(f"✅ Halo detectado: {avg_halo:.2f} mm ({len(measurements)} mediciones)")

            elif 'G21' in norma or 'ASTM_G21' in norma:
                print(f"\n{'='*60}")
                print(f" PROCESANDO IMAGEN PARA ASTM G21")
                print(f"   Microorganismo: {microorg_selec}")
                print(f"{'='*60}\n")
                
                rating, coverage_percentage, fungal_mask = analyzer.analyze_fungal_growth(
                    orig_img, ms, microorganism=microorg_selec
                )
                
                if fungal_mask is not None and np.sum(fungal_mask > 0) > 0:
                    overlay = orig_img.copy()
                    
                    # Determinar color según microorganismo
                    micro_lower = microorg_selec.lower()
                    
                    if 'aspergillus' in micro_lower or 'niger' in micro_lower:
                        color = [180, 50, 50]  # Rojo oscuro
                        print("🔴 Aplicando overlay ROJO para Aspergillus")
                    elif 'trichoderma' in micro_lower:
                        color = [0, 255, 100]  # Verde brillante
                        print("🟢 Aplicando overlay VERDE para Trichoderma")
                    else:
                        color = [0, 255, 0]
                        print("⚪ Aplicando overlay VERDE genérico")
                    
                    # Crear overlay con 50% de transparencia para mejor visibilidad
                    overlay[fungal_mask > 0] = color
                    processed_img = cv2.addWeighted(orig_img, 0.5, overlay, 0.5, 0)
                    
                    print(f"✅ Overlay aplicado: {np.sum(fungal_mask > 0)} píxeles coloreados")
                    print(f"   Cobertura: {coverage_percentage:.2f}%")
                    print(f"   Rating: {rating}/4\n")
                else:
                    processed_img = orig_img.copy()
                    print("⚠️ No se detectó crecimiento fúngico\n")        

            elif 'JIS' in norma or 'Z2801' in norma:
                treated_count, treated_colonies = analyzer.count_colonies_opencv(orig_img, ms)
                processed_img = orig_img.copy()
                cv2.drawContours(processed_img, treated_colonies, -1, (255, 0, 0), 2)
            elif 'E1428' in norma:
                coverage_percentage, colored_img, has_growth = analyzer.analyze_streptomyces_growth(orig_img, ms)

                processed_img = colored_img

                results.update({
                    "coverage_percentage": round(coverage_percentage, 2),
                    "has_visible_growth": has_growth,
                    "material_resistance": "No resistente" if has_growth else "Resistente",
                    "interpretation": "Presencia de crecimiento" if has_growth else "Ausencia de crecimiento"
                })


            # Guardar en lista
            treated_results_list.append({
                'original': orig_img,
                'pca': pca,
                'meanshift': ms,
                'processed': processed_img,
                'results': results
            })

            # Mostrar imágenes y métricas (tu código actual)
            st.markdown(f"### Réplica tratada {idx+1}")
            cols = st.columns(2)

           # Verificar y mostrar la imagen
            if orig_img is not None and isinstance(orig_img, np.ndarray):
                
                # *** CORRECCIÓN CLAVE: BGR a RGB y ELIMINACIÓN de 'use_container_width' ***
                try:
                    # 1. Convertir de BGR (OpenCV) a RGB (Streamlit/PIL)
                    # Esto es necesario ya que el debugg mostró que es un NumPy array (OpenCV).
                    if len(orig_img.shape) == 3 and orig_img.shape[2] == 3:
                        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    else:
                        orig_img_rgb = orig_img # Ya es escala de grises o 4 canales
                        
                    # 2. Mostrar la imagen. ELIMINAMOS 'use_container_width' para evitar el TypeError
                    cols[0].image(orig_img_rgb, caption="Original") 
                
                except Exception as e:
                    # Fallback en caso de que la conversión falle por alguna razón inesperada
                    st.error(f"❌ Error al intentar mostrar la imagen. Detalles: {str(e)}")
                    # Mostrar el original sin ningún argumento problemático
                    cols[0].image(orig_img, caption="Original (Fallo en Conversión/Muestra)")

            else:
                # Manejo del caso donde orig_img es None
                cols[0].error(f"❌ Error: No se pudo cargar o procesar la imagen original de la réplica {idx+1}.")

            # Si también está usando use_container_width=True para la segunda columna (processed_img), 
            # debe eliminarlo allí también para mantener la consistencia.
            if processed_img is not None:
                cols[1].image(processed_img, caption="Resultado final") # AQUI TAMBIÉN DEBE ELIMINAR 'use_container_width'        
            # Mostrar métricas según norma
            if 'AATCC' in norma or 'TM147' in norma:
                halo = results.get('inhibition_halo_mm', 0)
                presencia = results.get('has_inhibition', False)
                c1, c2, c3 = st.columns(3)
                c1.metric("Halo (mm)", f"{halo:.2f}")
                c2.metric("Presencia de inhibición", "Sí" if presencia else "No")
                color = "🟢" if presencia else "🔴"
                c3.markdown(f"**Interpretación:** {color}")

            elif 'G21' in norma:
                cobertura = results.get('coverage_percentage', 0)
                rating = results.get('astm_g21_rating', 'No disponible')
                c1, c2, c3 = st.columns(3)
                c1.metric("Cobertura (%)", f"{cobertura:.2f}")
                c2.metric("Rating", rating)
                c3.markdown(f"**Interpretación:** {results.get('interpretation', '')}")
            elif 'JIS' in norma or 'Z2801' in norma:
                treated_count = results.get('treated_count', 0)
                log_red = results.get('log_reduction', 0)
                c1, c2 = st.columns(2)
                c1.metric("Colonias TRATADA", treated_count)
                if isinstance(log_red, (int, float)):
                    c2.metric("Reducción logarítmica", f"{log_red:.2f}")
                else:
                    c2.metric("Reducción logarítmica", str(log_red))
            elif 'E1428' in norma:
                c1, c2, c3 = st.columns(3)
                c1.metric("Cobertura (%)", f"{results.get('coverage_percentage', 0):.2f}")
                c2.metric("Crecimiento visible", "Sí" if results.get('has_visible_growth', False) else "No")
                c3.markdown(f"**Interpretación:** {results.get('interpretation', '')}")
            
            st.markdown("---")

        #  AHORA SÍ: EXTRAER VALORES DESPUÉS DEL LOOP
        print("\n" + "="*60)
        print(" EXTRAYENDO VALORES DE RÉPLICAS")
        print(f"Total de réplicas en lista: {len(treated_results_list)}")
        print("="*60)

        valores_tratadas = []  #  Inicializar AQUÍ
        for idx, replica_data in enumerate(treated_results_list):
            results = replica_data['results']
            
            # Determinar qué valor extraer según la norma
            valor_extraido = None
            if "AATCC" in norma or "TM147" in norma:
                valor_extraido = results.get("inhibition_halo_mm", None)
            elif "JIS" in norma or "Z2801" in norma:
                valor_extraido = results.get("log_reduction", None)
            elif "ASTM_G21" in norma or "G21" in norma:
                valor_extraido = results.get("coverage_percentage", None)
            elif "ASTM_E1428" in norma or "E1428" in norma:
                valor_extraido = results.get("coverage_percentage", None)
            
            print(f"Réplica {idx+1}: {valor_extraido} (tipo: {type(valor_extraido)})")
            
            # Solo agregar si es numérico válido
            if valor_extraido is not None and isinstance(valor_extraido, (int, float)):
                valores_tratadas.append(float(valor_extraido))
                print(f"   Agregado: {valor_extraido}")
            else:
                print(f"   Rechazado (no numérico)")

        print(f"\n Lista final: {valores_tratadas}")
        print(f"Total valores válidos: {len(valores_tratadas)}")
        print("="*60 + "\n")

        #  CALCULAR ESTADÍSTICAS
        media = 0.0
        desviacion = 0.0

        if valores_tratadas and len(valores_tratadas) > 0:
            media = float(np.mean(valores_tratadas))
            
            if len(valores_tratadas) > 1:
                desviacion = float(np.std(valores_tratadas, ddof=1))
            else:
                desviacion = 0.0
            
            print(f" Media: {media:.2f}")
            print(f" Desviación: {desviacion:.2f}")
        else:
            print(" No hay valores para calcular estadísticas")

        # Guardar en session_state
        st.session_state["num_replicas"] = len(treated_results_list)
        st.session_state["valores_replicas"] = valores_tratadas
        st.session_state["media"] = media
        st.session_state["desviacion"] = desviacion
        st.session_state["treated_results_list"] = treated_results_list

        if treated_results_list:
            last_result = treated_results_list[-1]
            st.session_state["results"] = last_result['results']
            st.session_state["original_img"] = last_result['original']
            st.session_state["processed_img"] = last_result['processed']

        if control_results_list:
            st.session_state["control_results_list"] = control_results_list
            st.session_state["control_img_rgb"] = control_results_list[0]['original']
            st.session_state["control_processed"] = control_results_list[0]['processed']

        # Mostrar en interfaz
        if valores_tratadas and len(valores_tratadas) > 1:
            st.success(" Estadísticas finales de todas las réplicas tratadas")
            c1, c2, c3 = st.columns(3)
            c1.metric("Número de réplicas", len(valores_tratadas))
            c2.metric("Promedio", f"{media:.2f}")
            c3.metric("Desviación estándar", f"{desviacion:.2f}")
        elif valores_tratadas and len(valores_tratadas) == 1:
            st.info(" Solo 1 réplica analizada")
            c1, c2 = st.columns(2)
            c1.metric("Número de réplicas", 1)
            c2.metric("Valor", f"{media:.2f}")
        else:
            st.warning(" No se encontraron valores numéricos válidos para calcular estadísticas")       

            # ============= ANÁLISIS ESTADÍSTICO CON TEST T =============
        # Funciona con 1 o más réplicas por grupo

        test_t_results = None

        # CASO 1: JIS Z 2801 (Control vs Tratada)
        if norma == "JIS Z 2801 2010" and control_results_list and treated_results_list:
            valores_control_ttest = [c['count'] for c in control_results_list]
            valores_tratadas_ttest = valores_tratadas
            
            test_t_results = realizar_test_t_flexible(
                valores_control_ttest, 
                valores_tratadas_ttest,
                nombre_grupo1="Control",
                nombre_grupo2="Tratada"
            )

        # Guardar resultados
        if test_t_results:
            st.session_state["test_t_results"] = test_t_results
            
            # MOSTRAR RESULTADOS (funciona para cualquier cantidad de réplicas)
            if test_t_results.get('suficientes_datos', False):
                st.markdown("---")
                
                tipo_analisis = test_t_results.get('tipo_analisis', 'completo')
                
                # ENCABEZADO según tipo de análisis
                if tipo_analisis == 'completo':
                    st.markdown("##  Análisis Estadístico: Test T de Student")
                    st.info(" **Análisis estadístico completo disponible** (≥2 réplicas por grupo)")
                elif tipo_analisis == 'simple':
                    st.markdown("##  Comparación de Resultados")
                    st.warning(" **Análisis descriptivo únicamente** (1 réplica por grupo). Para Test T estadístico se requieren al menos 2 réplicas por grupo.")
                elif tipo_analisis == 'limitado':
                    st.markdown("##  Comparación de Resultados")
                    st.warning(" **Análisis limitado** (al menos un grupo tiene solo 1 réplica). Para Test T completo se requieren ≥2 réplicas en ambos grupos.")
                
                # MOSTRAR MÉTRICAS según tipo de análisis
                if tipo_analisis == 'completo':
                    # Métricas del Test T completo
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Estadístico t", f"{test_t_results['t_statistic']:.4f}")
                    
                    with col2:
                        st.metric("Valor p", f"{test_t_results['p_value']:.4f}")
                    
                    with col3:
                        st.metric("Grados de libertad", test_t_results['grados_libertad'])
                    
                    with col4:
                        if test_t_results['es_significativo']:
                            st.markdown("**Resultado:** 🟢")
                            st.markdown("**Significativo**")
                        else:
                            st.markdown("**Resultado:** 🔴")
                            st.markdown("**No significativo**")
                    
                    # Tabla comparativa completa
                    st.markdown("###  Estadísticas Descriptivas por Grupo")
                    data_ttest = {
                        'Grupo': [test_t_results['nombre_grupo1'], test_t_results['nombre_grupo2']],
                        'n (réplicas)': [test_t_results['n_grupo1'], test_t_results['n_grupo2']],
                        'Media': [f"{test_t_results['media_grupo1']:.2f}", 
                                f"{test_t_results['media_grupo2']:.2f}"],
                        'Desv. Estándar': [f"{test_t_results['std_grupo1']:.2f}", 
                                        f"{test_t_results['std_grupo2']:.2f}"]
                    }
                    df_ttest = pd.DataFrame(data_ttest)
                    st.table(df_ttest)
                    
                    # Interpretación
                    st.success(f"** Interpretación:** {test_t_results['interpretacion']}")
                    
                    # Información adicional
                    ci_lower, ci_upper = test_t_results['intervalo_confianza']
                    st.markdown(f"""
                    **Diferencia de medias:** {test_t_results['diferencia_medias']:.2f}  
                    **Intervalo de confianza 95%:** [{ci_lower:.2f}, {ci_upper:.2f}]
                    """)
                    
                    # Explicación del valor p
                    with st.expander(" ¿Qué significa el valor p?"):
                        st.markdown('''
                        - **p < 0.05**: La diferencia ES estadísticamente significativa
                        - **p ≥ 0.05**: La diferencia NO ES estadísticamente significativa
                        
                        **Nota:** Un resultado significativo indica que es muy improbable 
                        que la diferencia observada se deba al azar.
                        ''')
                
                else:  # Análisis simple o limitado
                    # Métricas básicas
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(f"{test_t_results['nombre_grupo1']}", 
                                f"{test_t_results['media_grupo1']:.2f}",
                                delta=None)
                        st.caption(f"n = {test_t_results['n_grupo1']}")
                    
                    with col2:
                        st.metric(f"{test_t_results['nombre_grupo2']}", 
                                f"{test_t_results['media_grupo2']:.2f}",
                                delta=f"{-test_t_results['diferencia_medias']:.2f}")
                        st.caption(f"n = {test_t_results['n_grupo2']}")
                    
                    with col3:
                        st.metric("Diferencia", 
                                f"{test_t_results['diferencia_medias']:.2f}")
                        if 'porcentaje_reduccion' in test_t_results:
                            st.caption(f"Reducción: {test_t_results['porcentaje_reduccion']:.1f}%")
                    
                    # Tabla simplificada
                    st.markdown("###  Resumen de Datos")
                    data_simple = {
                        'Grupo': [test_t_results['nombre_grupo1'], test_t_results['nombre_grupo2'], 'Diferencia'],
                        'n': [test_t_results['n_grupo1'], test_t_results['n_grupo2'], '-'],
                        'Valor': [f"{test_t_results['media_grupo1']:.2f}", 
                                f"{test_t_results['media_grupo2']:.2f}",
                                f"{test_t_results['diferencia_medias']:.2f}"]
                    }
                    
                    # Agregar desviación si está disponible
                    if test_t_results.get('std_grupo1') is not None or test_t_results.get('std_grupo2') is not None:
                        data_simple['Desv. Estándar'] = [
                            f"{test_t_results['std_grupo1']:.2f}" if test_t_results.get('std_grupo1') is not None else 'N/A',
                            f"{test_t_results['std_grupo2']:.2f}" if test_t_results.get('std_grupo2') is not None else 'N/A',
                            '-'
                        ]
                    
                    df_simple = pd.DataFrame(data_simple)
                    st.table(df_simple)
                    
                    # Interpretación descriptiva
                    st.info(f"** Observación:** {test_t_results['interpretacion']}")
                    
                    # Recomendación
                    st.markdown("""
                    <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                        <h4 style="margin-top: 0; color: #856404;">💡 Recomendación</h4>
                        <p style="margin-bottom: 0; color: #856404;">
                        Para obtener un análisis estadístico robusto con Test T, se recomienda realizar 
                        <strong>al menos 2-3 réplicas por grupo</strong>. Esto permitirá:
                        </p>
                        <ul style="color: #856404; margin-bottom: 0;">
                            <li>Calcular la variabilidad de las mediciones</li>
                            <li>Determinar si las diferencias son estadísticamente significativas</li>
                            <li>Obtener intervalos de confianza</li>
                            <li>Mayor confiabilidad en las conclusiones</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f" {test_t_results.get('error', 'No se pudo realizar el análisis')}")

        # CASO 2: Otras normas - comparar réplicas tratadas contra valor de referencia
        # (Solo si tienes un valor de referencia conocido)
        elif len(treated_results_list) >= 2:
            # Ejemplo: Comparar tus réplicas contra un estándar conocido
            # valores_estandar = [valor_referencia] * len(valores_tratadas)  # Descomentar si tienes referencia
            # test_t_results = realizar_test_t_flexible(valores_tratadas, valores_estandar, "Muestra", "Estándar")
            pass

        # Guardar resultados
        if test_t_results:
            st.session_state["test_t_results"] = test_t_results
            
            # MOSTRAR RESULTADOS DEL TEST T
            if test_t_results.get('suficientes_datos', False):
                st.markdown("---")
                st.markdown("##  Análisis Estadístico: Test T de Student")
                
                st.info("**Nota:** Este análisis determina si existe una diferencia estadísticamente significativa entre los grupos analizados (α = 0.05)")
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Estadístico t", f"{test_t_results['t_statistic']:.4f}")
                
                with col2:
                    st.metric("Valor p", f"{test_t_results['p_value']:.4f}")
                
                with col3:
                    st.metric("Grados de libertad", test_t_results['grados_libertad'])
                
                with col4:
                    if test_t_results['es_significativo']:
                        st.markdown("**Resultado:** 🟢")
                        st.markdown("**Significativo**")
                    else:
                        st.markdown("**Resultado:** 🔴")
                        st.markdown("**No significativo**")
                
                # Tabla comparativa
                st.markdown("###  Estadísticas Descriptivas por Grupo")
                data_ttest = {
                    'Grupo': [test_t_results['nombre_grupo1'], test_t_results['nombre_grupo2']],
                    'n (réplicas)': [test_t_results['n_grupo1'], test_t_results['n_grupo2']],
                    'Media': [f"{test_t_results['media_grupo1']:.2f}", 
                            f"{test_t_results['media_grupo2']:.2f}"],
                    'Desv. Estándar': [f"{test_t_results['std_grupo1']:.2f}", 
                                    f"{test_t_results['std_grupo2']:.2f}"]
                }
                df_ttest = pd.DataFrame(data_ttest)
                st.table(df_ttest)
                
                # Interpretación destacada
                st.success(f"** Interpretación:** {test_t_results['interpretacion']}")
                
                # Información adicional
                ci_lower, ci_upper = test_t_results['intervalo_confianza']
                st.markdown(f"""
                 **Diferencia de medias:** {test_t_results['diferencia_medias']:.2f}  
                **Intervalo de confianza 95%:** [{ci_lower:.2f}, {ci_upper:.2f}]
                """)
                
                # Explicación del valor p
                with st.expander(" ¿Qué significa el valor p?"):
                    st.markdown('''
                    - **p < 0.05**: La diferencia ES estadísticamente significativa (rechazamos la hipótesis nula)
                    - **p ≥ 0.05**: La diferencia NO ES estadísticamente significativa (no rechazamos la hipótesis nula)
                    
                    **Nota:** Un resultado significativo no siempre implica relevancia práctica. 
                    Siempre considera el contexto del ensayo.
                    ''')
            else:
                st.warning(f" {test_t_results.get('error', 'No se pudo realizar el Test T')}")
                st.info(" **Sugerencia:** Sube al menos 2 réplicas de cada grupo para habilitar el análisis estadístico.")

        
        # FUNCIÓN PARA MOSTRAR EN EL REPORTE
        def mostrar_test_t_en_reporte():
            """
            Muestra los resultados del Test T en el reporte.
            SE INSERTA DESPUÉS DE LA GRÁFICA DE MEDIA CON ERROR ESTÁNDAR
            """
            test_t_results = st.session_state.get("test_t_results", None)
            
            if not test_t_results or not test_t_results.get('suficientes_datos', False):
                return  # No mostrar si no hay datos
            
            st.markdown("---")
            st.markdown("##  Análisis Estadístico: Test T de Student")
            
            st.markdown("""
                <div class="reporte-section">
                    <p><strong>Objetivo:</strong> Determinar si existe una diferencia estadísticamente significativa 
                    entre los grupos analizados con un nivel de confianza del 95% (α = 0.05).</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Métricas principales en cuadros
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-box">
                        <p style="color: #666; margin-bottom: 5px; font-size: 14px;">Estadístico t</p>
                        <h2 style="color: #667eea; margin: 0;">{test_t_results['t_statistic']:.4f}</h2>
                        <p style="color: #999; font-size: 11px; margin-top: 5px;">df = {test_t_results['grados_libertad']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-box">
                        <p style="color: #666; margin-bottom: 5px; font-size: 14px;">Valor p</p>
                        <h2 style="color: #667eea; margin: 0;">{test_t_results['p_value']:.4f}</h2>
                        <p style="color: #999; font-size: 11px; margin-top: 5px;">α = 0.05</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                color = "#4caf50" if test_t_results['es_significativo'] else "#f44336"
                significancia = "SÍ" if test_t_results['es_significativo'] else "NO"
                icono = "✓" if test_t_results['es_significativo'] else "✗"
                st.markdown(f"""
                    <div class="metric-box">
                        <p style="color: #666; margin-bottom: 5px; font-size: 14px;">¿Significativo?</p>
                        <h1 style="color: {color}; margin: 0; font-size: 48px;">{icono}</h1>
                        <p style="color: {color}; font-size: 14px; margin-top: 5px; font-weight: bold;">{significancia}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Tabla de estadísticas descriptivas
            st.markdown("### Estadísticas Descriptivas")
            data = {
                'Grupo': [test_t_results['nombre_grupo1'], test_t_results['nombre_grupo2'], 'Diferencia'],
                'n': [
                    test_t_results['n_grupo1'], 
                    test_t_results['n_grupo2'],
                    '-'
                ],
                'Media': [
                    f"{test_t_results['media_grupo1']:.2f}",
                    f"{test_t_results['media_grupo2']:.2f}",
                    f"{test_t_results['diferencia_medias']:.2f}"
                ],
                'Desv. Estándar': [
                    f"{test_t_results['std_grupo1']:.2f}",
                    f"{test_t_results['std_grupo2']:.2f}",
                    '-'
                ]
            }
            df = pd.DataFrame(data)
            st.table(df)
            
            # Gráfica de comparación con significancia
            st.markdown("###  Visualización de la Comparación")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            grupos = [test_t_results['nombre_grupo1'], test_t_results['nombre_grupo2']]
            medias = [test_t_results['media_grupo1'], test_t_results['media_grupo2']]
            errores = [test_t_results['std_grupo1'], test_t_results['std_grupo2']]
            
            bars = ax.bar(grupos, medias, yerr=errores, capsize=12, 
                        color=['#667eea', '#4facfe'], 
                        edgecolor='black', linewidth=2, alpha=0.85)
            
            # Valores sobre las barras
            for bar, media in zip(bars, medias):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{media:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=13)
            
            # Línea de significancia si aplica
            if test_t_results['es_significativo']:
                y_max = max([m + e for m, e in zip(medias, errores)])
                y_line = y_max * 1.15
                
                # Línea horizontal
                ax.plot([0, 1], [y_line, y_line], 'k-', linewidth=2.5)
                # Líneas verticales
                ax.plot([0, 0], [medias[0] + errores[0], y_line], 'k-', linewidth=2.5)
                ax.plot([1, 1], [medias[1] + errores[1], y_line], 'k-', linewidth=2.5)
                
                # Texto de significancia
                ax.text(0.5, y_line + y_max*0.03, 
                    f'p = {test_t_results["p_value"]:.4f} *', 
                    ha='center', va='bottom', fontweight='bold', fontsize=13,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
            
            ax.set_ylabel('Media ± Desviación Estándar', fontsize=13, fontweight='bold')
            ax.set_title(f'Comparación: {test_t_results["nombre_grupo1"]} vs {test_t_results["nombre_grupo2"]}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
            ax.set_ylim(bottom=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.caption("* Diferencia estadísticamente significativa (p < 0.05)")
            
            # Interpretación destacada
            ci_lower, ci_upper = test_t_results['intervalo_confianza']
            st.markdown(f"""
                <div class="conclusion-box">
                    <h4> Interpretación Estadística</h4>
                    <p style="font-size: 15px; line-height: 1.6;">{test_t_results['interpretacion']}</p>
                    <hr style="border: 1px solid #ddd; margin: 15px 0;">
                    <p><strong>Intervalo de confianza 95% para la diferencia de medias:</strong> 
                    [{ci_lower:.2f}, {ci_upper:.2f}]</p>
                    <p><strong>Grados de libertad:</strong> {test_t_results['grados_libertad']}</p>
                    <p style="margin-top: 10px; font-size: 13px; color: #666;">
                    <em>Nota: El intervalo de confianza indica el rango probable de la verdadera diferencia 
                    entre las medias poblacionales con un 95% de confianza.</em>
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # --- BOTÓN DE REPORTE (FUERA DEL BLOQUE DE ANÁLISIS) ---
    # Este botón se muestra si ya hay resultados guardados en session_state
    if "treated_results_list" in st.session_state and st.session_state.get("treated_results_list"):
        st.markdown("---")
        st.markdown("###  Generar reporte del ensayo")
        st.success(" Resultados disponibles para reporte")
        
        # Botón para ir a reporte
        if st.button(" Ver reporte completo", key="btn_ver_reporte"):  # <--- clave única
            st.session_state["pagina"] = "reporte"
            st.rerun()
            
    # --- PÁGINA DE REPORTE ---

elif st.session_state["pagina"] == "reporte":
    # CSS personalizado para el reporte
    st.markdown("""
        <style>
        .reporte-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            color: white;
            margin-bottom: 30px;
            text-align: center;
        }
        .reporte-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }
        .metric-box {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .conclusion-box {
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
            margin-top: 20px;
            text-align: justify;  
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Recuperar o crear analyzer ---
    if "analyzer" not in st.session_state:
        st.session_state["analyzer"] = MultiStandardAnalyzer()
    analyzer = st.session_state["analyzer"]
    
    
    # Botón para volver
    col_back, col_space = st.columns([1, 5])
    with col_back:
        if st.button("⬅ Volver", key="btn_volver_reporte"):  # <--- clave única
            st.session_state["pagina"] = "parametros"
            st.rerun()
    
    # Recuperar información del estado
    norma = st.session_state.get("norma", "No especificada")
    microorg_selec = st.session_state.get("microorg_selec", "No definido")
    medio = st.session_state.get("medio", "")
    tiempo = st.session_state.get("tiempo", "")
    temperatura = st.session_state.get("temperatura", "")
    dilucion = st.session_state.get("dilucion", "")
    escala = st.session_state.get("escala", "")

    # DEFINIR mm_per_pixel AQUÍ (ANTES DE USARLO)
    mm_per_pixel = float(escala) if escala else 0.05

    results = st.session_state.get("results", {})
    imagen_tratada = st.session_state.get("original_img", None)      
    
    
    # ENCABEZADO DEL REPORTE
   
    st.markdown("""
        <div class="reporte-header">
            <h1> REPORTE DE ANÁLISIS MICROBIOLÓGICO</h1>
            <p style="font-size: 18px; margin-top: 10px;">MicroVision</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================
    # SECCIÓN 1: INFORMACIÓN DEL ENSAYO (IMAGEN + DESCRIPCIÓN)
    # ========================================
    st.markdown("##  1. INFORMACIÓN DEL ENSAYO")

    # Verificar si es JIS
    norma = st.session_state.get("norma", "No especificada")  # Recuperar norma primero
    es_jis = 'JIS' in str(norma) or 'Z2801' in str(norma)

    # --- SUBSECCIÓN: DESCRIPCIÓN DEL ENSAYO ---
    st.markdown("""
        <div class="reporte-section">
            <h3> Descripción del ensayo</h3>
        </div>
    """, unsafe_allow_html=True)

    descripcion = generar_descripcion(norma, microorg_selec, medio, tiempo, temperatura, escala, dilucion)
    st.markdown(descripcion, unsafe_allow_html=True)

    st.markdown("---")

    # --- SUBSECCIÓN: MUESTRAS ANALIZADAS ---
    st.markdown("""
        <div class="reporte-section">
            <h3> Muestras analizadas</h3>
        </div>
    """, unsafe_allow_html=True)

    # === PARA NORMA JIS (control + tratadas) ===
    if es_jis:
        control_results_list = st.session_state.get("control_results_list", [])
        treated_results_list = st.session_state.get("treated_results_list", [])

        # ----- CONTROLES -----
        if control_results_list:
            st.markdown("### 🧫 Muestras CONTROL")
            num_control = len(control_results_list)
            for i in range(0, num_control, 3):
                cols = st.columns(min(3, num_control - i))
                for j, col in enumerate(cols):
                    idx = i + j
                    replica = control_results_list[idx]
                    with col:
                        st.image(
                            replica['original'],
                            use_container_width=True,
                            caption=f"Control {idx+1} | Colonias: {replica.get('count', 'N/A')}"
                        )

        # ----- TRATADAS -----
        if treated_results_list:
            st.markdown("### 🧫 Muestras TRATADAS")
            num_tratadas = len(treated_results_list)
            for i in range(0, num_tratadas, 3):
                cols = st.columns(min(3, num_tratadas - i))
                for j, col in enumerate(cols):
                    idx = i + j
                    replica = treated_results_list[idx]
                    count = replica.get('results', {}).get('treated_count', 'N/A')
                    with col:
                        st.image(
                            replica['original'],
                            use_container_width=True,
                            caption=f"Tratada {idx+1} | Colonias: {count}"
                        )

    # === PARA OTRAS NORMAS (solo tratadas) ===
    else:
        treated_results_list = st.session_state.get("treated_results_list", [])
        if treated_results_list:
            st.markdown("### 🧫 Muestras analizadas")
            num_imgs = len(treated_results_list)
            for i in range(0, num_imgs, 3):
                cols = st.columns(min(3, num_imgs - i))
                for j, col in enumerate(cols):
                    idx = i + j
                    replica = treated_results_list[idx]
                    with col:
                        st.image(
                            replica['original'],
                            use_container_width=True,
                            caption=f"Réplica {idx+1}"
                        )
        else:
            st.warning("No hay imágenes disponibles.")

         # SECCIÓN 2: RESULTADOS 
        st.markdown("##  2. RESULTADOS DEL ANÁLISIS")

        if es_jis:
                # RESULTADOS DETALLADOS PARA JIS (CONTROL + TRATADA)
            
                
                control_results_list = st.session_state.get("control_results_list", [])
                treated_results_list = st.session_state.get("treated_results_list", [])
                
                #  RESULTADOS CONTROL 
                if control_results_list:
                    st.markdown("###  Resultados de Réplicas CONTROL")
                    
                    # Tabla resumen de controles
                    data_control = {
                        'Réplica': [f"Control {i+1}" for i in range(len(control_results_list))],
                        'Colonias Detectadas (UFC)': [c['count'] for c in control_results_list]
                    }
                    df_control = pd.DataFrame(data_control)
                    st.table(df_control)
                    
                    # Estadísticas del control
                    valores_control = [c['count'] for c in control_results_list]
                    media_control = np.mean(valores_control)
                    std_control = np.std(valores_control, ddof=1) if len(valores_control) > 1 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Media Control", f"{media_control:.2f} UFC")
                    col2.metric("Desv. Estándar", f"{std_control:.2f}" if len(valores_control) > 1 else "N/A")
                    col3.metric("Total Réplicas", len(valores_control))
                    
                    st.markdown("---")
                
                # RESULTADOS TRATADAS 
                if treated_results_list:
                    st.markdown("###  Resultados de Réplicas TRATADAS")
                    
                    # Tabla resumen de tratadas
                    data_tratadas = []
                    for idx, replica in enumerate(treated_results_list):
                        results = replica['results']
                        data_tratadas.append({
                            'Réplica': f"Tratada {idx+1}",
                            'Colonias Detectadas (UFC)': results.get('treated_count', 0),
                            'Reducción Log': f"{results.get('log_reduction', 'N/A'):.2f}" if isinstance(results.get('log_reduction'), (int, float)) else 'N/A'
                        })
                    
                    df_tratadas = pd.DataFrame(data_tratadas)
                    st.table(df_tratadas)
                    
                    # Estadísticas de las tratadas
                    valores_tratadas = [r['results'].get('treated_count', 0) for r in treated_results_list]
                    media_tratadas = np.mean(valores_tratadas)
                    std_tratadas = np.std(valores_tratadas, ddof=1) if len(valores_tratadas) > 1 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Media Tratada", f"{media_tratadas:.2f} UFC")
                    col2.metric("Desv. Estándar", f"{std_tratadas:.2f}" if len(valores_tratadas) > 1 else "N/A")
                    col3.metric("Total Réplicas", len(valores_tratadas))
                    
                    st.markdown("---")
                
                # ========== COMPARACIÓN CONTROL VS TRATADA ==========
                if control_results_list and treated_results_list:
                    st.markdown("###  Comparación General: Control vs Tratada")
                    
                    # Calcular reducción logarítmica promedio
                    log_reductions = []
                    for replica in treated_results_list:
                        log_red = replica['results'].get('log_reduction')
                        if isinstance(log_red, (int, float)):
                            log_reductions.append(log_red)
                    
                    if log_reductions:
                        log_red_promedio = np.mean(log_reductions)
                        log_red_std = np.std(log_reductions, ddof=1) if len(log_reductions) > 1 else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Control Promedio", f"{media_control:.2f} UFC")
                        
                        with col2:
                            delta_val = -(media_control - media_tratadas)
                            st.metric("Tratada Promedio", f"{media_tratadas:.2f} UFC", delta=f"{delta_val:.2f}")
                        
                        with col3:
                            if log_red_promedio >= 3.0:
                                color = "🟢"
                            elif log_red_promedio >= 2.0:
                                color = "🟡"
                            elif log_red_promedio >= 1.0:
                                color = "🟠"
                            else:
                                color = "🔴"
                            st.markdown(f"**Reducción Log Promedio:** {color}")
                            st.markdown(f"### {log_red_promedio:.2f} ± {log_red_std:.2f}")
                        
                        with col4:
                            porcentaje_reduccion = ((media_control - media_tratadas) / media_control * 100) if media_control > 0 else 0
                            st.metric("% Reducción", f"{porcentaje_reduccion:.1f}%")
                        
                        # Interpretación
                        if log_red_promedio >= 2.0:
                            st.success(f" **CUMPLE** con el criterio JIS Z 2801 (R ≥ 2). Actividad antimicrobiana efectiva.")
                        else:
                            st.error(f" **NO CUMPLE** con el criterio JIS Z 2801 (R ≥ 2). Actividad antimicrobiana insuficiente.")

        else:
               
                # RESULTADOS PARA OTRAS NORMAS
            
                
            treated_results_list = st.session_state.get("treated_results_list", [])
            
            if treated_results_list:
                for i, replica in enumerate(treated_results_list):
                    results = replica["results"]
                    norma_res = results.get("standard", "")
                    st.markdown(f"###  Réplica tratada {i+1}")
                    st.markdown("<hr style='border:1px solid #bbb;'>", unsafe_allow_html=True)

                    if "AATCC" in norma_res or "TM147" in norma_res:
                        col1, col2, col3 = st.columns(3)
                        halo = results.get("inhibition_halo_mm", 0)
                        presencia = results.get("has_inhibition", False)
                        interpretacion = results.get("interpretation", "No efectivo")

                        with col1:
                            st.metric("Halo de inhibición", f"{halo:.2f} mm")

                        with col2:
                            st.metric("Inhibición detectada", "Sí" if presencia else "No")

                        with col3:
                            color = "🟢" if presencia else "🔴"
                            st.markdown(f"**Resultado:** {color}")
                            st.markdown(f"**{interpretacion}**")

                    elif "ASTM_G21" in norma_res or "G21" in norma_res:
                        col1, col2, col3 = st.columns(3)
                        cobertura = results.get("coverage_percentage", 0)
                        rating = results.get("astm_g21_rating", 0)

                        # Interpretación corta según el rating ASTM G21
                        if rating == 0:
                            color = "🟢"
                            interpretacion = "Sin crecimiento fúngico"
                        elif rating <= 2:
                            color = "🟡"
                            interpretacion = "Crecimiento moderado"
                        else:
                            color = "🔴"
                            interpretacion = "Crecimiento alto"

                        with col1:
                            st.metric("Cobertura fúngica", f"{cobertura:.2f}%")

                        with col2:
                            st.metric("Rating ASTM G21", f"{rating}/4")

                        with col3:
                            st.markdown(f"**Clasificación:** {color}")
                            st.markdown(f"**{interpretacion}**")
                    elif "ASTM_E1428" in norma_res or "E1428" in norma_res:
                        col1, col2, col3 = st.columns(3)
                        count = results.get("colony_count", 0)
                        growth = results.get("has_visible_growth", False)
                        resistencia = results.get("material_resistance", "")

                        with col1:
                            st.metric("Colonias detectadas", count)

                        with col2:
                            st.metric("Crecimiento visible", "Sí" if growth else "No")

                        with col3:
                            color = "🔴" if growth else "🟢"
                            st.markdown(f"**Estado:** {color}")
                            st.markdown(f"**{resistencia}**")

                    st.markdown("<hr style='border:0.5px solid #ccc;'>", unsafe_allow_html=True)
            else:
                st.warning("No hay resultados disponibles para mostrar.")
                
            # GRÁFICA 
            st.markdown("---")
            st.markdown("###  Representación gráfica")

            # Recuperar imágenes subidas desde session_state
            imagenes_subidas = st.session_state.get("imagenes_subidas", {})
            tratadas = imagenes_subidas.get("tratada", [])
            controles = imagenes_subidas.get("control", [])

            # Mostrar gráfica SOLO si hay exactamente una imagen tratada (y control si aplica)
            if norma == "JIS Z 2801 2010":
                if len(controles) == 1 and len(tratadas) == 1:
                    plot_results_by_norm(norma, results)
            elif len(tratadas) == 1:
                plot_results_by_norm(norma, results)
            # No mostrar mensaje si hay varias réplicas

                        
    # Mostrar estadísticas globales solo si NO es JIS 
    if "jis" not in norma.lower() and "z2801" not in norma.lower():

        #  Estadísticas globales 
        st.markdown("---")
        st.markdown("##  Estadísticas globales")

        #  Recuperar variables del session_state
        num_replicas = st.session_state.get("num_replicas", 0)
        valores_tratadas = st.session_state.get("valores_replicas", [])
        media = st.session_state.get("media", 0)
        desviacion = st.session_state.get("desviacion", 0)

        # Mostrar métricas principales
        col1, col2, col3 = st.columns(3)
        col1.metric("Réplicas analizadas", num_replicas)
        col2.metric("Promedio", f"{media:.2f}")
        col3.metric("Desviación estándar", f"{desviacion:.2f}")

        #  Gráfica con media y error estándar 
        if valores_tratadas and len(valores_tratadas) > 1:
            #st.markdown("### Gráfica")

            # Reemplazar valores negativos por 0
            valores_tratadas = [max(0, v) for v in valores_tratadas]

            # Calcular error estándar (desviación / √n)
            n = len(valores_tratadas)
            error_estandar = desviacion / (n ** 0.5)

            # Crear figura
            fig, ax = plt.subplots(figsize=(4, 5))
            ax.bar(1, media, yerr=error_estandar, capsize=10, color="#667eea",
                edgecolor="black", width=0.4, label="Media ± Error estándar")

            ax.set_xticks([1])
            ax.set_xticklabels([""])
            ax.set_ylabel("Media")
            ax.set_title("")

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No hay suficientes réplicas para generar la gráfica de error estándar.")

   
    # SECCIÓN 3: CONCLUSIÓN 
    st.markdown("---")
    st.markdown("##  3. CONCLUSIÓN")

    interpretacion = ""

    # Ver cuántas imágenes tratadas hay
    treated_results_list = st.session_state.get("treated_results_list", [])
    num_replicas = len(treated_results_list)
    media = st.session_state.get("media", 0)
    desviacion = st.session_state.get("desviacion", 0)

    # ✅ CASO 1: SOLO UNA IMAGEN
    if num_replicas == 1:
        if norma and "AATCC" in str(norma):
            halo = results.get("inhibition_halo_mm", 0)
            if halo > 0:
                interpretacion = (
                    f"el material evaluado presentó un halo de inhibición de {halo:.2f} mm frente a "
                    f"*{microorg_selec}*. este resultado indica actividad antibacteriana según la norma "
                    f"{norma}. la presencia del halo de inhibición demuestra que el material tiene "
                    f"propiedades que inhiben el crecimiento bacteriano en su entorno inmediato."
                ).lower()
            else:
                interpretacion = (
                    f"no se observó halo de inhibición frente a *{microorg_selec}*, indicando ausencia de actividad antibacteriana "
                    f"según la norma {norma}. el material no presenta propiedades que inhiban el crecimiento bacteriano."
                ).lower()

        elif norma and "ASTM G21" in str(norma):
            rating = results.get("astm_g21_rating", None)
            coverage = results.get("coverage_percentage", 0)
            escala_texto = {
                0: "resistente a hongos (ningún crecimiento visible)",
                1: "resistencia parcial (crecimiento escaso < 10%)",
                2: "resistencia limitada (crecimiento ligero 10-30%)",
                3: "baja resistencia (crecimiento moderado 30-60%)",
                4: "no resistente (crecimiento abundante > 60%)"
            }
            if rating is not None:
                interpretacion = (
                    f"el material obtuvo una calificación de {rating} en la escala astm g21-15, "
                    f"con una cobertura fúngica del {coverage:.2f}%. esto indica que el material es: "
                    f"{escala_texto.get(rating, 'valor no definido')}. "
                )
                if rating <= 1:
                    interpretacion += "el material muestra excelente resistencia al ataque fúngico."
                elif rating == 2:
                    interpretacion += "el material presenta resistencia moderada, aceptable en ciertas aplicaciones."
                else:
                    interpretacion += "el material presenta baja resistencia, susceptible al deterioro por hongos."
                interpretacion = interpretacion.lower()

        elif norma and "JIS" in str(norma):
            logR = results.get("log_reduction", 0)
            control_c = results.get("control_count", 0)
            treated_c = results.get("treated_count", 0)
            cumple = "cumple" if logR >= 2 else "no cumple"

            interpretacion = (
                f"el material presentó una reducción logarítmica de {logR:.2f} frente a *{microorg_selec}*, "
                f"con {control_c} colonias en el control y {treated_c} colonias en la tratada. "
                f"por lo tanto, el material {cumple} con la norma {norma}."
            ).lower()

        elif norma and "ASTM E1428" in str(norma):
            growth = results.get("has_visible_growth", False)
            coverage = results.get("coverage_percentage", 0)
            if growth:
                interpretacion = (
                    f"se observó crecimiento visible con {coverage:.2f}% de cobertura. "
                    f"según la norma {norma}, el material no es resistente al ataque de actinomicetos (*{microorg_selec}*)."
                ).lower()
            else:
                interpretacion = (
                    f"no se observó crecimiento visible significativo ({coverage:.2f}% de cobertura). "
                    f"el material es resistente al ataque de actinomicetos según la norma {norma}."
                ).lower()

    # ✅ CASO 2: VARIAS RÉPLICAS
    elif num_replicas > 1:
        if "ASTM G21" in str(norma):
            interpretacion = (
                f"se analizaron {num_replicas} réplicas del ensayo según la norma astm g21-15. "
                f"el material presentó una cobertura fúngica promedio de {media:.2f}% ± {desviacion:.2f}% (de). "
            )
            if media <= 10:
                interpretacion += "esto indica que el material posee excelente resistencia al crecimiento fúngico."
            elif media <= 30:
                interpretacion += "esto indica que el material presenta resistencia moderada frente al ataque de hongos."
            elif media <= 60:
                interpretacion += "el material muestra baja resistencia, con crecimiento fúngico apreciable."
            else:
                interpretacion += "el material es no resistente, presentando una alta cobertura por hongos."
            interpretacion = interpretacion.lower()
        
        elif "AATCC" in str(norma):
            interpretacion = (
                f"se analizaron {num_replicas} réplicas según la norma {norma}, con un halo de inhibición promedio de {media:.2f} ± {desviacion:.2f} mm (de). "
            )
            if media > 0:
                interpretacion += f"el material muestra actividad antibacteriana efectiva frente a *{microorg_selec}*."
            else:
                interpretacion += "no se detectó halo de inhibición significativo, indicando ausencia de actividad antibacteriana."
            interpretacion = interpretacion.lower()

        elif "JIS" in str(norma):
            interpretacion = (
                f"el análisis de {num_replicas} réplicas según la norma {norma} mostró una reducción logarítmica promedio de {media:.2f} ± {desviacion:.2f} (de). "
            )
            if media >= 2:
                interpretacion += "el material cumple con los criterios de eficacia antimicrobiana (r ≥ 2)."
            else:
                interpretacion += "el material no cumple con los criterios mínimos de eficacia antimicrobiana."
            interpretacion = interpretacion.lower()

        elif "ASTM E1428" in str(norma):
            interpretacion = (
                f"se analizaron {num_replicas} réplicas según la norma astm e1428, obteniendo una cobertura promedio de {media:.2f}% ± {desviacion:.2f}% (de). "
            )
            if media <= 10:
                interpretacion += "el material es resistente al ataque de actinomicetos."
            else:
                interpretacion += "el material presenta susceptibilidad moderada o alta al biodeterioro."
            interpretacion = interpretacion.lower()

    else:
        interpretacion = "no se detectaron resultados válidos para generar una conclusión.".lower()

    # Convertir a HTML limpio (sin cambiar el case ya que ya está en minúsculas)
    interpretacion_html = interpretacion.replace('**', '<strong>').replace('*', '<em>').replace('\n\n', '<br><br>')

    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 10px; text-align: justify;'>
    {interpretacion_html}
    </div>
    """, unsafe_allow_html=True)

    # PIE DE PÁGINA
    
    st.markdown("---")
    from datetime import datetime  # asegúrate de tener esta línea al inicio
    fecha_actual = datetime.now().strftime("%d de %B de %Y, %H:%M:%S")



    st.markdown(f"""
        <div style="text-align: center; color: #666; padding: 20px; font-size: 12px;">
            <p><strong>MicroVision - Sistema de Análisis Microbiológico Automatizado</strong></p>
            <p>Reporte generado automáticamente el {fecha_actual}</p>
            <p>Pontificia Universidad Javeriana</p>
        </div>
    """, unsafe_allow_html=True)

    # ==================== BLOQUE DE DESCARGA DE PDF ====================
    import streamlit as st
    from datetime import datetime

    # Guardar toda la información necesaria en session_state antes de generar el PDF
    st.session_state["descripcion"] = generar_descripcion(
        st.session_state.get("norma", ""),
        st.session_state.get("microorg_selec", ""),
        st.session_state.get("medio", ""),
        st.session_state.get("tiempo", ""),
        st.session_state.get("temperatura", ""),
        st.session_state.get("escala", ""),
        st.session_state.get("dilucion", "")
    )

    # Interpretación para conclusión
    # Aquí asumimos que ya generaste 'interpretacion_html' como en tu código
    # Si no, la generamos igual que antes
    if "interpretacion" not in st.session_state or not st.session_state["interpretacion"]:
        st.session_state["interpretacion"] = interpretacion_html if "interpretacion_html" in locals() else "No disponible"

    # Guardar datos de análisis
    st.session_state["treated_results_list"] = st.session_state.get("treated_results_list", [])
    st.session_state["control_results_list"] = st.session_state.get("control_results_list", [])
    st.session_state["valores_replicas"] = st.session_state.get("valores_replicas", [])
    st.session_state["media"] = st.session_state.get("media", 0)
    st.session_state["desviacion"] = st.session_state.get("desviacion", 0)

    # Botón para generar PDF
    st.markdown("---")
    st.markdown("### 📄 Descargar Reporte")

    if st.button("📥 Descargar reporte completo en PDF", type="primary", use_container_width=True):
        with st.spinner("Generando PDF profesional... ⏳"):
            try:
                pdf_buffer = generar_pdf_reporte_completo()
                
                fecha_archivo = datetime.now().strftime('%Y%m%d_%H%M%S')
                nombre_archivo = f"Reporte_MicroVision_{fecha_archivo}.pdf"
                
                st.download_button(
                    label="⬇️ Descargar PDF",
                    data=pdf_buffer,
                    file_name=nombre_archivo,
                    mime="application/pdf",
                    key="download_pdf",
                    use_container_width=True
                )
                
                st.success("✅ PDF generado correctamente")
                st.info("💡 El PDF incluye todas las secciones del reporte con imágenes, tablas y conclusión")
            
            except Exception as e:
                st.error(f"❌ Error al generar PDF: {str(e)}")
                import traceback
                with st.expander("🔍 Ver detalles del error"):
                    st.code(traceback.format_exc())

