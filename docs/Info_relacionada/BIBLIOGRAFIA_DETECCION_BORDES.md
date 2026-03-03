# Bibliografia: Deteccion de Bordes de Pagina en Documentos Historicos Digitalizados

**Documento preparado para el equipo de desarrollo del pipeline Copista**
**Fecha:** Marzo 2026
**Contexto:** Revision bibliografica orientada a la deteccion del poligono (cuadrilatero) que delimita el borde fisico del papel en fotografias de paginas de periodico historico (corpus "Heraldo"). Los principales desafios son: papeles u objetos que sobresalen de los margenes, paginas adyacentes visibles, y dificultad para detectar el borde inferior (lomo/pliegue de encuadernacion).

---

## 1. Lista de Papers Relevantes

### 1.1 Deteccion de bordes y marcos de pagina (metodos clasicos y semi-clasicos)

---

**Paper 1**

**Titulo:** Document cleanup using page frame detection

**Autores y ano:** Shafait, F., van Beusekom, J., Keysers, D., Breuel, T.M. (2010)

**Fuente:** International Journal on Document Analysis and Recognition (IJDAR), Springer, Vol. 13, No. 1, pp. 43-52

**DOI/URL:** https://doi.org/10.1007/s10032-008-0071-7 | https://www.researchgate.net/publication/220163614

**Relevancia:** Es uno de los trabajos fundacionales en deteccion de marco de pagina. Propone un algoritmo de matching geometrico para encontrar el marco optimo de pagina en documentos estructurados (articulos de journal, libros, revistas) explotando la propiedad de alineacion del texto. Directamente aplicable al corpus Heraldo ya que aborda la remocion de texto de paginas vecinas (ruido textual) y bordes negros (ruido no textual). Reporta una reduccion del error de OCR del 4.3% al 1.7% en el dataset UW-III al remover los caracteres fuera del marco detectado.

---

**Paper 2**

**Titulo:** Border detection of document images scanned from large books

**Autores y ano:** Shamqoli, A., Khosravi, H. (2013/2014)

**Fuente:** Iranian Conference on Machine Vision and Image Processing (MVIP). Tambien disponible en IEEE Xplore.

**DOI/URL:** https://ieeexplore.ieee.org/document/6779955 | https://www.researchgate.net/publication/264461394

**Relevancia:** Trabajo especificamente orientado al problema de libros grandes digitalizados, que es muy analogo al corpus Heraldo. La imagen resultante frecuentemente tiene un borde oscuro y regiones de texto de la pagina vecina. Propone una tecnica novedosa basada en perfiles de proyeccion combinados con deteccion de bordes. El paper existente en el directorio del proyecto (`Shamqoli-Unknown-BorderDetectionofDocumentImagesScannedFromLargeBooks.pdf`) es justamente este trabajo, por lo que es referencia directa.

---

**Paper 3**

**Titulo:** Automatic Borders Detection of Camera Document Images

**Autores y ano:** Stamatopoulos, N., Gatos, B., Kesidis, A.L. (2007)

**Fuente:** Proceedings of the Second International Workshop on Camera-Based Document Analysis and Recognition (CBDAR), Curitiba, Brasil

**URL:** https://users.iit.demokritos.gr/~bgat/CBDAR_BORDERS.pdf | https://www.semanticscholar.org/paper/Automatic-Borders-Detection-of-Camera-Document-Stamatopoulos-Gatos/a9d81a16843cd4b734ad7acbba3306bda4488c13

**Relevancia:** Propone una tecnica para imagenes de documentos capturadas con camara digital (no escaner plano), que es el caso del corpus Heraldo. La metodologia combina perfiles de proyeccion con un proceso de deteccion de bordes para identificar y eliminar bordes negros ruidosos y texto de paginas vecinas. Evaluado en el dataset DFKI-I (CBDAR 2007 Dewarping Contest), lo que lo hace directamente comparable con otros trabajos.

---

**Paper 4**

**Titulo:** Border Noise Removal of Camera-Captured Document Images Using Page Frame Detection

**Autores y ano:** Bukhari, S.S., Shafait, F., Breuel, T.M. (2012)

**Fuente:** In: Iwamura, M., Shafait, F. (eds) Camera-Based Document Analysis and Recognition. CBDAR 2011. Lecture Notes in Computer Science, vol. 7139. Springer, Berlin, Heidelberg.

**DOI/URL:** https://link.springer.com/chapter/10.1007/978-3-642-29364-1_10 | https://www.researchgate.net/publication/228517530

**Relevancia:** Extiende el trabajo de Shafait et al. (2010) para imagenes capturadas con camara (en lugar de escaner), que es el caso exacto del corpus Heraldo. Utiliza informacion de contenido textual y no-textual para encontrar el marco de pagina del documento. Evaluado en el dataset DFKI-I (CBDAR 2007). Aborda explicitamente los dos tipos de ruido marginal: ruido textual (texto de la pagina contigua) y ruido no-textual (superficie de la mesa, fondo, etc.).

---

**Paper 5**

**Titulo:** A Robust Page Frame Detection Method for Complex Historical Document Images

**Autores y ano:** (Autores del Departamento de Ciencias de la Computacion, Universidad de Kaiserslautern) (2018)

**Fuente:** Publicado en ResearchGate; orientado a documentos historicos del siglo XVI-XIX.

**URL:** https://www.researchgate.net/publication/329760609_A_Robust_Page_Frame_Detection_Method_for_Complex_Historical_Document_Images

**Relevancia:** Este paper aborda especificamente la deteccion de marcos de pagina en documentos historicos complejos con mala calidad de imagen, caracteres danados y grandes cantidades de ruido textual y no-textual, que es exactamente el escenario del corpus Heraldo. Usa transformadas morfologicas, el detector de segmentos de linea (LSD) y un algoritmo de matching geometrico. Reporta un incremento del 4.49% en precision de OCR para documentos historicos del siglo XVI-XIX, y del 6.69% para documentos contemporaneos.

---

**Paper 6**

**Titulo:** Page frame detection for double page document images

**Autores y ano:** Yan, H., et al. (aproximadamente 2009-2012)

**Fuente:** Disponible en ResearchGate

**URL:** https://www.researchgate.net/publication/220933023_Page_frame_detection_for_double_page_document_images

**Relevancia:** Aborda el caso de escaneo de paginas dobles de un libro (dos paginas en una imagen), que es un subproblema del corpus Heraldo: la aparicion de la pagina adyacente en los bordes. Propone un algoritmo para detectar los marcos de las dos paginas y separarlas, usando proyecciones de corridas blancas verticales y horizontales. Altamente relevante para el problema de paginas adyacentes visibles en el corpus.

---

**Paper 7**

**Titulo:** Page Frame Detection for Marginal Noise Removal from Scanned Documents

**Autores y ano:** Shafait, F., et al. (2007-2008)

**Fuente:** Publicado en ResearchGate; trabajo relacionado con el paper de IJDAR 2010.

**URL:** https://www.researchgate.net/publication/220809267_Page_Frame_Detection_for_Marginal_Noise_Removal_from_Scanned_Documents

**Relevancia:** Trabajo temprano del mismo grupo que establece las bases para la deteccion de marco de pagina como tecnica de eliminacion de ruido marginal. Propone el uso de perfiles de proyeccion para detectar zonas de ruido en los bordes del documento escaneado. Complementa al paper de IJDAR 2010 y es util para entender la evolucion de las tecnicas clasicas.

---

### 1.2 Deteccion de bordes basada en aprendizaje profundo

---

**Paper 8**

**Titulo:** PageNet: Page Boundary Extraction in Historical Handwritten Documents

**Autores y ano:** Tensmeyer, C., Davis, B.L., Wigington, C., Lee, I., Barrett, B. (2017)

**Fuente:** Proceedings of the 4th International Workshop on Historical Document Imaging and Processing (HIP 2017), Kyoto, Japon. Co-publicado por Adobe Research.

**DOI/URL:** https://arxiv.org/abs/1709.01618 | https://research.adobe.com/publication/pagenet-page-boundary-extraction-in-historical-handwritten-documents/ | https://www.researchgate.net/publication/322241186

**Relevancia:** Sistema de aprendizaje profundo que identifica la region principal de la pagina en una imagen para segmentar el contenido del ruido del borde. Usa una Red Completamente Convolucional (FCN) para segmentacion pixel a pixel, post-procesada en una region cuadrilatera. Obtuvo mas del 94% de mean intersection over union en 4 colecciones de documentos historicos manuscritos. Crucialmente, puede segmentar documentos que estan superpuestos sobre otros documentos, lo que es directamente relevante para las paginas adyacentes del corpus Heraldo.

---

**Paper 9**

**Titulo:** dhSegment: A generic deep-learning approach for document segmentation

**Autores y ano:** Oliveira, S.A., Seguin, B., Kaplan, F. (2018)

**Fuente:** 2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR), IEEE, pp. 7-12.

**DOI/URL:** https://arxiv.org/abs/1804.10371 | https://doi.org/10.1109/icfhr-2018.2018.00011 | https://github.com/dhlab-epfl/dhSegment

**Relevancia:** Framework de codigo abierto del DHLAB de la EPFL que ofrece una solucion generica para segmentacion de documentos historicos. Aborda multiples tareas simultaniamente: extraccion de pagina, extraccion de baselines, analisis de layout. Usa una arquitectura CNN para prediccion pixel a pixel con bloques de post-procesamiento. Creado originalmente para el proyecto "Venice Time Machine" y tiene implementacion publica en Python, lo que lo hace directamente utilizable para el corpus Heraldo.

---

**Paper 10**

**Titulo:** Fully Convolutional Neural Networks for Page Segmentation of Historical Document Images

**Autores y ano:** Wick, C., Puppe, F. (2018)

**Fuente:** 2018 13th IAPR International Workshop on Document Analysis Systems (DAS), IEEE. DOI: 10.1109/DAS.2018.8395210. Preprint: arXiv:1711.07695

**DOI/URL:** https://arxiv.org/abs/1711.07695 | https://ieeexplore.ieee.org/document/8395210/ | https://www.researchgate.net/publication/321210733

**Relevancia:** Propone una FCN de alto rendimiento para segmentacion de documentos historicos disenada para procesar una pagina completa en un solo paso, aprendiendo directamente desde pixels sin preprocesamiento. Introduce una metrica novedosa (Foreground Pixel Accuracy, FgPA) que solo cuenta pixels de primer plano para evaluacion mas representativa. Altamente relevante porque el corpus Heraldo requiere justamente procesar paginas completas de documentos historicos con degradacion.

---

**Paper 11**

**Titulo:** Approach for Document Detection by Contours and Contrasts

**Autores y ano:** Skoryukina, N., et al. (2020)

**Fuente:** arXiv:2008.02615v2, Computer Vision and Pattern Recognition (cs.CV)

**DOI/URL:** https://arxiv.org/abs/2008.02615 | https://www.researchgate.net/publication/343498941

**Relevancia:** Propone una modificacion del metodo clasico basado en contornos donde las hipotesis competidoras de localizacion del contorno se clasifican segun el contraste entre las areas dentro y fuera del borde. Aborda explicitamente los casos de oclusion, fondo complejo y desenfoque. Logra una disminucion del 40% en errores de ordenamiento de alternativas y 10% de reduccion en el numero total de errores de deteccion. Evaluado en el dataset MIDV-500, que es un benchmark estandar para este tipo de tarea.

---

**Paper 12**

**Titulo:** Fast and Accurate Document Detection for Scanning (Dropbox Engineering Blog)

**Autores y ano:** Equipo de Machine Learning de Dropbox (2019-2020)

**Fuente:** Blog de ingenieria de Dropbox (tech blog, implementacion industrial)

**URL:** https://dropbox.tech/machine-learning/fast-and-accurate-document-detection-for-scanning

**Relevancia:** Aunque es un blog tecnico y no un paper academico formal, describe una solucion de produccion real basada en DenseNet-121 para regresion de las coordenadas de las 4 esquinas del documento (cuadrilatero). Incluye lecciones aprendidas de escala industrial: el sistema usa una CNN para detectar las esquinas del documento y esta entrenado con cientos de imagenes anotadas manualmente. Muy relevante porque el corpus Heraldo tiene exactamente el mismo objetivo: detectar las 4 esquinas de la pagina del periodico.

---

**Paper 13**

**Titulo:** Deep Learning for Historical Document Analysis and Recognition -- A Survey

**Autores y ano:** Simistira Liwicki, F., Seuret, M., Eichenberger, N., Garz, A., Liwicki, M., Ingold, R. (2021)

**Fuente:** Journal of Imaging, MDPI, Vol. 6, No. 10, p. 110. Tambien en PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8321201/

**DOI/URL:** https://www.mdpi.com/2313-433X/6/10/110 | https://doi.org/10.3390/jimaging6100110

**Relevancia:** Survey comprehensivo sobre el estado del arte en analisis de documentos historicos con aprendizaje profundo. Cubre metodos para deteccion de bordes, segmentacion de paginas, analisis de layout, y OCR en documentos historicos. Sirve como punto de entrada para entender la evolucion desde metodos clasicos (Canny, Hough) hacia FCN y transformers. Identifica los principales datasets de benchmark y las metricas de evaluacion estandar para este tipo de tarea.

---

**Paper 14**

**Titulo:** Holistically-Nested Edge Detection

**Autores y ano:** Xie, S., Tu, Z. (2015, extendido 2017)

**Fuente:** Proceedings of the IEEE International Conference on Computer Vision (ICCV 2015), pp. 1395-1403. DOI: 10.1109/ICCV.2015.164

**DOI/URL:** https://arxiv.org/abs/1504.06375 | https://ieeexplore.ieee.org/document/7410521/ | https://github.com/s9xie/hed

**Relevancia:** HED es un detector de bordes basado en aprendizaje profundo que supera significativamente a los metodos clasicos (Canny). Usa una red VGGNet con supervision profunda en multiples escalas para producir mapas de bordes precisos. En el contexto del corpus Heraldo, HED puede detectar los bordes del papel en condiciones donde Canny falla (iluminacion variable, degradacion, fondo complejo). Tiene implementacion directa en OpenCV con el modulo `cv2.dnn`. ODS=0.782 en BSDS500, superando a Canny (ODS=0.600).

---

**Paper 15**

**Titulo:** Handheld Video Document Scanning: A Robust On-Device Model for Multi-Page Document Scanning

**Autores y ano:** (Autores no especificados en busqueda) (2024)

**Fuente:** arXiv:2411.00576, noviembre 2024

**DOI/URL:** https://arxiv.org/abs/2411.00576

**Relevancia:** Trabajo muy reciente (2024) que aborda el escaneo de multiples paginas de documentos desde video en dispositivo movil, incluyendo deteccion automatica de esquinas y bordes de pagina. Propone un modelo robusto que puede manejar movimiento y cambios de pagina. Aunque orientado a smartphones, las tecnicas de deteccion de bordes en presencia de paginas multiples y oclusiones son directamente transferibles al corpus Heraldo.

---

**Paper 16**

**Titulo:** DocUNet: Document Image Unwarping via a Stacked U-Net

**Autores y ano:** Ma, K., Shu, Z., et al. (2018)

**Fuente:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018), IEEE. DOI: 10.1109/CVPR.2018.00490

**DOI/URL:** https://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_DocUNet_Document_Image_CVPR_2018_paper.pdf | https://ieeexplore.ieee.org/document/8578592/

**Relevancia:** Aunque DocUNet esta orientado al dewarping (corrector de curvatura), el preprocesamiento necesario para entrenar y usar este sistema incluye deteccion del borde de la pagina. Es el primer metodo basado en aprendizaje profundo para desenrollar imagenes de documentos, y establece el dataset DocUNet que es ampliamente usado como benchmark. Relevante para el problema del lomo/pliegue del corpus Heraldo, que es una forma de distorsion que DocUNet aborda.

---

**Paper 17**

**Titulo:** Document Layout Analysis: A Comprehensive Survey

**Autores y ano:** Binmakhashen, G.M., Mahmoud, S.A. (2019)

**Fuente:** ACM Computing Surveys, Vol. 52, No. 6, Article 103, pp. 1-36. DOI: 10.1145/3355610

**DOI/URL:** https://dl.acm.org/doi/10.1145/3355610

**Relevancia:** Survey comprehensivo que cubre metodos clasicos y modernos para analisis de layout de documentos, incluyendo deteccion de bordes de pagina. Proporciona un marco taxonomico para clasificar los diferentes enfoques: basados en textura, morfologicos, proyeccion de perfiles, aprendizaje automatico. Util como referencia sistematica de las tecnicas disponibles y sus condiciones de aplicabilidad para el equipo de desarrollo del pipeline Copista.

---

**Paper 18**

**Titulo:** DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks

**Autores y ano:** Das, S., Ma, K., Shu, Z., Samaras, D., Shilkrot, R. (2019)

**Fuente:** Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019), pp. 131-140. DOI: 10.1109/ICCV.2019.00022

**DOI/URL:** https://openaccess.thecvf.com/content_ICCV_2019/html/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_Networks_ICCV_2019_paper.html | https://github.com/cvlab-stonybrook/DewarpNet

**Relevancia:** DewarpNet modela explicitamente la forma 3D del documento (incluyendo curvatura del lomo) para corregir la distorsion. El dataset Doc3D generado contiene anotaciones de forma 3D, normales de superficie, mapa UV, e imagen de albedo. El problema del lomo/pliegue de encuadernacion que afecta al corpus Heraldo es exactamente el caso que DewarpNet aborda. Reduce el error de caracter en OCR un 42% en promedio. Codigo disponible en GitHub.

---

**Paper 19**

**Titulo:** Page Layout Analysis System for Unconstrained Historic Documents

**Autores y ano:** Quiroga, F., et al. (2021)

**Fuente:** Document Analysis and Recognition - ICDAR 2021. Lecture Notes in Computer Science, vol. 12824. Springer, Cham. DOI: 10.1007/978-3-030-86331-9_32

**DOI/URL:** https://link.springer.com/chapter/10.1007/978-3-030-86331-9_32 | https://dl.acm.org/doi/10.1007/978-3-030-86331-9_32

**Relevancia:** Sistema de analisis de layout para documentos historicos sin restricciones, que aborda la gran variabilidad de formatos en documentos historicos. Especialmente relevante porque los documentos del corpus Heraldo (periodico historico) tienen multiples columnas, diferentes tamanios de fuente y calidad de papel variable, que son exactamente los desafios mencionados en este paper.

---

**Paper 20**

**Titulo:** DocScanner: Robust Document Image Rectification with Progressive Learning

**Autores y ano:** Feng, H., et al. (2021/2025)

**Fuente:** arXiv:2110.14968. Publicado en International Journal of Computer Vision (IJCV), 2025. DOI: 10.1007/s11263-025-02431-5

**DOI/URL:** https://arxiv.org/abs/2110.14968 | https://github.com/fh2019ustc/DocScanner

**Relevancia:** DocScanner introduce un mecanismo de aprendizaje progresivo con arquitectura recurrente para corregir imagenes de documentos, manteniendo una estimacion unica que se corrige progresivamente. Robusto ante condiciones no controladas de captura (posicion de camara, deformaciones fisicas, variaciones de iluminacion). El aprendizaje progresivo es especialmente util para el corpus Heraldo donde las condiciones de captura son heterogeneas.

---

## 2. Catalogo de Tecnicas

### Tecnicas Clasicas

| Tecnica | Papers que la usan | Descripcion breve |
|---|---|---|
| **Perfiles de Proyeccion (Projection Profiles)** | Shamqoli & Khosravi (2013), Stamatopoulos & Gatos (2007), Shafait et al. (2010), Paper 5, Paper 6 | Calcula la suma de pixeles negros/blancos por fila/columna para detectar transiciones de borde. Simple y rapido, util cuando el fondo es claramente distinto al documento. |
| **Deteccion de bordes Canny** | Shamqoli & Khosravi (2013), Stamatopoulos & Gatos (2007), pipeline Copista actual | Detector de bordes multi-etapa: suavizado Gaussian, gradiente de Sobel, supresion de no-maximos, histéresis de umbral. Sensible a ruido y parametros; falla con iluminacion no uniforme. |
| **Transformada de Hough** | Pipeline Copista actual, varios trabajos clasicos | Transforma el espacio de imagen al espacio de parametros de linea. Cada pixel de borde "vota" por las lineas que pasan por el. Las lineas con mas votos son las fronteras del documento. |
| **Componentes Conectadas (Connected Components)** | Bukhari et al. (2012), Shafait et al. (2010) | Analisis de regiones conectadas en imagen binarizada para separar el documento del fondo y del ruido de pagina adyacente. Disponible como `cv2.connectedComponentsWithStats()` en OpenCV. |
| **Algoritmo Flood Fill** | Varios trabajos de limpieza de documento | Rellena regiones contiguas desde una semilla. Usado para aislar el fondo de la pagina. Disponible como `cv2.floodFill()` en OpenCV. |
| **Morfologia Matematica (erosion, dilatacion, gradiente morfologico)** | Paper 5, Bukhari et al. (2012), varios | Operaciones sobre imagen binarizada para reforzar bordes y eliminar ruido. El gradiente morfologico (dilatacion - erosion) enfatiza los bordes. Disponible en OpenCV: `cv2.morphologyEx()`. |
| **Detector de Segmentos de Linea (LSD)** | Paper 5 (Kaiserslautern) | Algoritmo para deteccion de segmentos de linea rectos en imagen. Mas robusto que Hough para imagenes con ruido. Disponible en OpenCV como `cv2.createLineSegmentDetector()`. |
| **Correlacion cruzada de perfiles** | Shafait et al. (2010) | Verifica la correlacion entre los perfiles de proyeccion de la region de borde y un modelo esperado de ruido textual para confirmar si el borde detectado corresponde a una pagina adyacente. |
| **Approximacion poligonal de contornos** | Metodo clasico de OpenCV | `cv2.approxPolyDP()` aproxima un contorno a un poligono con N vertices. Para N=4 produce el cuadrilatero del documento. Requiere que el contorno este correctamente segmentado. |
| **Transformacion de perspectiva (warpPerspective)** | Metodos de escaneo movil | `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()` aplica la transformacion de perspectiva usando las 4 esquinas detectadas del documento para rectificarlo. |
| **GrabCut** | Usos en pipeline de escaneo de documentos | Segmentacion interactiva foreground/background basada en grafos de corte. Puede usarse con un rectangulo inicial alrededor del documento. `cv2.grabCut()` en OpenCV. |
| **Active Contours / Snakes** | SnakeCut (Prakash et al., 2007) | Contorno deformable que minimiza energia. Evoluciona hacia los bordes de la imagen bajo la influencia de fuerzas externas (gradiente) e internas (tension del contorno). Disponible en scikit-image. |
| **Watershed** | Usos generales en segmentacion de documentos | Trata la imagen como un mapa topografico. Con marcadores adecuados (foreground, background) separa regiones. `cv2.watershed()` en OpenCV. Util para separar documento del fondo. |

### Tecnicas Basadas en Aprendizaje Profundo

| Tecnica | Papers que la usan | Descripcion breve |
|---|---|---|
| **FCN (Fully Convolutional Network) para segmentacion pixel a pixel** | PageNet (Tensmeyer et al., 2017), Wick & Puppe (2018), dhSegment (Oliveira et al., 2018) | Red neuronal completamente convolucional que produce un mapa de segmentacion de la misma dimension que la entrada. Cada pixel se clasifica como "pagina" o "fondo". Post-procesamiento geometrico extrae el cuadrilatero. |
| **U-Net encoder-decoder con skip connections** | DocUNet (Ma et al., 2018), varios trabajos de segmentacion | Arquitectura simetrica con codificador (downsampling) y decodificador (upsampling) con conexiones directas entre capas equivalentes. Excelente para segmentacion con pocos datos de entrenamiento. |
| **Regresion de 4 esquinas (Corner Regression)** | Dropbox (2019), DenseNet-121 basado | Red CNN que directamente regresa las coordenadas (x,y) de las 4 esquinas del cuadrilatero del documento. Mas directa que la segmentacion, pero requiere supervision explicita de las esquinas. |
| **HED - Holistically-Nested Edge Detection** | Xie & Tu (2015/2017) | Red neuronal basada en VGGNet con supervision profunda en multiples salidas laterales para deteccion de bordes jerarquica. Produce mapas de bordes de alta calidad. Disponible via `cv2.dnn` en OpenCV. |
| **Segmentacion semantica pixel a pixel con encoder ResNet** | dhSegment (Oliveira et al., 2018), Wick & Puppe (2018) | Usa una red ResNet pre-entrenada como backbone del encoder. La transferencia de aprendizaje reduce los requerimientos de datos etiquetados. |
| **Estimacion 3D de forma + mapeo de textura** | DewarpNet (Das et al., 2019) | Dos sub-redes: una estima la forma 3D del documento (incluyendo curvatura del lomo), la otra aplica el mapa de textura para corregir la distorsion. Especialmente util para el problema del lomo. |
| **RANSAC para ajuste robusto de cuadrilatero** | Metodo clasico adaptado a deteccion de documentos | Random Sample Consensus: ajusta modelos geometricos (lineas, planos) en presencia de muchos outliers. Usado para filtrar las lineas de Hough y extraer el cuadrilatero optimo. `cv2.findHomography(method=cv2.RANSAC)`. |
| **Segmentacion + post-procesamiento geometrico** | PageNet (2017), dhSegment (2018) | Pipeline de dos etapas: (1) FCN produce mascara binaria de la region de pagina, (2) post-procesamiento geometrico extrae el cuadrilatero (convex hull, approxPolyDP, fitting de lineas). |
| **Deteccion de contraste borde/fondo** | Approach Contours & Contrasts (2020) | Clasifica hipotesis de borde segun el contraste entre las areas a los lados del contorno. El verdadero borde del documento tipicamente tiene alto contraste entre papel (claro) y fondo (oscuro). |
| **Transformers para rectificacion** | DocTr (Feng et al., 2021), DocScanner (2021) | Usan mecanismos de auto-atencion para capturar el contexto global de la imagen del documento y decodificar el desplazamiento pixel a pixel para corregir distorsion geometrica. |

---

## 3. Detalle de las Tecnicas Mas Prometedoras

### 3.1 Perfiles de Proyeccion + Deteccion de Bordes (Enfoque Clasico Mejorado)

**Descripcion del algoritmo:**

El metodo, utilizado en Shamqoli & Khosravi (2013) y Stamatopoulos & Gatos (2007), funciona de la siguiente manera:

1. **Preprocesamiento:** Conversion a escala de grises y binarizacion adaptativa (Otsu o Sauvola para documentos historicos).
2. **Perfiles de proyeccion horizontal:** Para cada fila de la imagen, se suma el numero de pixels negros. Las filas con muy pocos pixels negros corresponden a zonas de borde/margen.
3. **Perfiles de proyeccion vertical:** Idem para columnas. Identifica los limites izquierdo y derecho del area de texto.
4. **Deteccion de transiciones:** Se buscan las posiciones donde el perfil cambia de "borde" (pocos pixels negros) a "contenido" (muchos pixels negros). Estas son las coordenadas del marco de pagina.
5. **Refinamiento con deteccion de bordes:** Alrededor de las coordenadas detectadas por proyeccion, se aplica Canny o LSD para encontrar el borde exacto del papel.
6. **Correlacion cruzada:** Verifica si la zona de borde contiene texto de la pagina adyacente (ruido textual) o solo fondo (ruido no-textual).

**Parametros clave:**
- Umbral de binarizacion (manual, Otsu, Sauvola con ventana local)
- Umbral minimo de pixels negros para considerar una fila/columna como "contenido"
- Ancho de la zona de busqueda para refinamiento con Canny

**Fortalezas:**
- Muy rapido y sin requerimientos de GPU
- Implementable completamente en OpenCV/NumPy en pocas lineas
- Funciona bien cuando el margen entre el papel y el fondo es claramente visible
- No requiere datos de entrenamiento

**Debilidades:**
- Falla cuando papeles u objetos sobresalen de los margenes (oclusiones parciales)
- No maneja bien iluminacion no uniforme (sombras del lomo del libro)
- El borde inferior (lomo/pliegue) es dificil de detectar porque el gradiente de pixeles negros no es claro
- Si la pagina adyacente ocupa mucho espacio, el perfil de proyeccion no puede distinguirla claramente del contenido principal

**Condiciones de buen funcionamiento:**
- Fondo claramente mas oscuro o claro que el papel
- Documento sin objetos que sobresalgan de los margenes
- Iluminacion relativamente uniforme

**Implementacion en OpenCV:**

```python
import cv2
import numpy as np

def detectar_borde_proyeccion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarizacion adaptativa de Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perfil horizontal (suma de pixels negros por fila)
    h_profile = np.sum(binary == 0, axis=1)  # pixels oscuros por fila
    v_profile = np.sum(binary == 0, axis=0)  # pixels oscuros por columna

    # Umbral adaptativo: filas con menos del 5% del maximo
    umbral_h = 0.05 * h_profile.max()
    umbral_v = 0.05 * v_profile.max()

    # Encontrar primeras/ultimas filas con contenido real
    filas_contenido = np.where(h_profile > umbral_h)[0]
    cols_contenido = np.where(v_profile > umbral_v)[0]

    if len(filas_contenido) == 0 or len(cols_contenido) == 0:
        return None

    y_top = filas_contenido[0]
    y_bottom = filas_contenido[-1]
    x_left = cols_contenido[0]
    x_right = cols_contenido[-1]

    return np.array([[x_left, y_top], [x_right, y_top],
                     [x_right, y_bottom], [x_left, y_bottom]])
```

---

### 3.2 FCN / U-Net para Segmentacion Semantica de Pagina (Aprendizaje Profundo)

**Descripcion del algoritmo:**

Este es el enfoque de PageNet (2017) y dhSegment (2018). El pipeline tiene dos etapas:

**Etapa 1: Segmentacion por red neuronal**

1. Se toma la imagen completa (posiblemente redimensionada a 512x512 o 1024x1024) como entrada.
2. Una FCN (arquitectura encoder-decoder como U-Net o ResNet+FPN) produce un mapa de probabilidad pixel a pixel: probabilidad de que cada pixel pertenezca a la "pagina" o al "fondo".
3. Supervisada con mascaras de segmentacion binarias (pagina=1, fondo=0) generadas a partir de las anotaciones de poligono.

**Etapa 2: Post-procesamiento geometrico**

1. Binarizacion del mapa de probabilidad con umbral (ej. 0.5).
2. Operaciones morfologicas para cerrar huecos y eliminar ruido.
3. Encontrar el contorno mas grande (`cv2.findContours`).
4. Calcular el casco convexo (`cv2.convexHull`).
5. Aproximar a poligono de 4 puntos (`cv2.approxPolyDP`).

**Parametros clave:**
- Arquitectura del backbone (ResNet-50, ResNet-101, EfficientNet)
- Umbral de binarizacion del mapa de segmentacion
- Tolerancia de `approxPolyDP` (epsilon = porcentaje del perimetro, ej. 0.02)
- Tamano de kernel morfologico para cierre de huecos

**Fortalezas:**
- Muy robusto ante variaciones de iluminacion, degradacion y ruido
- Puede manejar oclusiones parciales si el entrenamiento incluye ejemplos de ese tipo
- PageNet demostro que puede segmentar documentos superpuestos sobre otros documentos
- Generaliza bien a diferentes tipos de documentos historicos
- No requiere parametros manuales delicados

**Debilidades:**
- Requiere datos de entrenamiento anotados (mascaras de segmentacion)
- Necesita GPU para entrenamiento (inferencia puede ser CPU si el modelo es pequeno)
- El post-procesamiento geometrico puede fallar si la mascara tiene muchos huecos
- Si el objeto que ocluye (papel que sobresale) es del mismo color que la pagina, la red puede incluirlo en la mascara

**Condiciones de buen funcionamiento:**
- Con datos de entrenamiento representativos del corpus Heraldo (imagenes similares)
- Cuando las oclusiones son parciales (no cubren mas del 30-40% del borde)
- Con imagenes de resolucion media-alta (mayor a 500x500 pixels)

**Implementacion con dhSegment (Python):**

dhSegment tiene implementacion en Python disponible en `https://github.com/dhlab-epfl/dhSegment`. El flujo de trabajo es:
1. Preparar dataset: pares de imagen + mascara binaria anotada.
2. Fine-tuning del modelo pre-entrenado con las imagenes del corpus Heraldo.
3. Inferencia: el modelo produce una mascara de probabilidad.
4. Post-procesamiento geometrico para extraer el cuadrilatero.

**Alternativa con PyTorch:**

```python
import torch
import torchvision
import cv2
import numpy as np

# Cargar modelo pre-entrenado (ejemplo con DeepLabV3)
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

def segmentar_pagina_deeplab(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Preprocesamiento
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img_rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    # Clase 0 = background, clase 15 = persona, etc.
    # Para documentos se necesita fine-tuning con clase "pagina"
    mask = output.argmax(0).byte().numpy()
    return mask

def extraer_cuadrilatero_de_mascara(mask):
    # Morfologia para limpiar
    kernel = np.ones((15, 15), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar contorno mas grande
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)

    # Casco convexo + aproximacion a 4 puntos
    hull = cv2.convexHull(largest)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    return approx
```

---

### 3.3 Deteccion por Contraste de Borde (Approach Contours & Contrasts)

**Descripcion del algoritmo:**

El metodo de Skoryukina et al. (arXiv:2008.02615, 2020) es una extension del enfoque clasico basado en contornos que agrega el criterio de contraste para seleccionar la hipotesis correcta:

1. **Deteccion de lineas candidatas:** Se detectan multiples lineas candidatas para cada uno de los 4 bordes del documento usando Hough o LSD. Esto genera muchas hipotesis.
2. **Formacion de cuadrilateros candidatos:** Se forman cuadrilateros combinando lineas candidatas para cada borde.
3. **Puntuacion por contraste:** Para cada cuadrilatero candidato, se calcula el contraste entre el area interior (pagina) y el area exterior (fondo). El cuadrilatero correcto deberia tener la mayor diferencia de contraste.
4. **Seleccion del mejor candidato:** El cuadrilatero con mayor puntuacion de contraste se selecciona como el borde de la pagina.
5. **Refinamiento iterativo:** Opcionalmente, se refina la posicion del borde.

**Parametros clave:**
- Numero de hipotesis de linea candidatas por borde
- Metrica de contraste (diferencia de luminosidad promedio, histograma, etc.)
- Criterio de seleccion del mejor cuadrilatero

**Fortalezas:**
- Mas robusto que el Hough puro ante oclusiones y fondos complejos
- No requiere entrenamiento
- Logro reduccion del 40% en errores de ordenamiento en MIDV-500
- La intuicion del contraste es directamente aplicable: el papel del periodico es mas claro que el fondo fotografico

**Debilidades:**
- Puede fallar cuando el fondo tiene contraste similar al papel (ej. mesas de color blanco)
- El contraste puede ser enganoso si la pagina adyacente (mas oscura, amarillenta) tiene contraste similar al fondo
- Requiere que al menos parte del borde sea visible (no funciona con oclusion total del borde)

**Condiciones de buen funcionamiento:**
- Cuando hay diferencia de color/luminosidad clara entre papel y fondo
- Cuando las oclusiones no cubren demasiado del borde
- Con imagenes sin desenfoque excesivo

**Implementacion simplificada en OpenCV:**

```python
import cv2
import numpy as np
from itertools import product

def contraste_cuadrilatero(img_gray, quad_pts):
    """Calcula contraste entre interior y exterior del cuadrilatero."""
    h, w = img_gray.shape
    mask_interior = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_interior, [quad_pts], 255)
    mask_exterior = cv2.bitwise_not(mask_interior)

    media_interior = img_gray[mask_interior > 0].mean()
    media_exterior = img_gray[mask_exterior > 0].mean()
    return abs(media_interior - media_exterior)

def detectar_borde_contraste(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Deteccion de bordes y lineas
    edges = cv2.Canny(img_blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                             minLineLength=50, maxLineGap=10)
    if lines is None:
        return None

    # Generar candidatos de cuadrilatero (simplificado)
    # En implementacion real se agrupan por orientacion
    mejor_cuadrilatero = None
    mejor_contraste = -1

    # [Logica de combinacion de lineas y evaluacion por contraste]
    # Este es el esquema; la implementacion completa requiere
    # clasificar lineas por orientacion (horizontal/vertical/diagonal)
    # y probar combinaciones.

    return mejor_cuadrilatero
```

---

### 3.4 Holistically-Nested Edge Detection (HED) como Preprocesador

**Descripcion del algoritmo:**

HED (Xie & Tu, ICCV 2015) usa una arquitectura VGGNet modificada con supervision profunda en 5 salidas laterales para producir mapas de bordes a multiples escalas. El resultado final es la fusion de todas las escalas.

**Uso en deteccion de bordes de pagina:**

HED no detecta el cuadrilatero directamente, sino que produce un mapa de bordes mucho mas limpio y robusto que Canny. Este mapa se usa como entrada para los pasos subsiguientes del pipeline (Hough, contornos, etc.):

1. **Aplicar HED** a la imagen: produce un mapa de bordes de alta calidad.
2. **Filtrar por magnitud:** umbralizar el mapa HED.
3. **Detectar lineas:** Hough o LSD sobre el mapa HED.
4. **Extraer cuadrilatero:** a partir de las lineas detectadas.

**Parametros clave:**
- Umbral de binarizacion del mapa HED (ej. 0.3 sobre el mapa de probabilidad)
- Parametros de Hough posteriores

**Fortalezas:**
- Detecta bordes en condiciones donde Canny falla: iluminacion no uniforme, bajo contraste, degradacion historica
- Los bordes son semanticamente significativos (el modelo aprende que bordes importan)
- Tiene implementacion directa en OpenCV via `cv2.dnn`
- El modelo pre-entrenado (disponible en GitHub) funciona razonablemente sin fine-tuning

**Debilidades:**
- Mas lento que Canny (requiere inferencia de red neuronal, ~400ms por imagen 320x480)
- Detecta TODOS los bordes, no solo el borde de la pagina: requiere post-procesamiento para aislar el borde externo
- El modelo pre-entrenado fue entrenado en BSDS500 (imagenes naturales), no en documentos historicos

**Condiciones de buen funcionamiento:**
- Con imagenes donde el borde del papel es visible pero de bajo contraste
- Como alternativa a Canny cuando el pipeline actual falla por ruido/iluminacion

**Implementacion con OpenCV:**

```python
import cv2
import numpy as np

def aplicar_HED(img_bgr, prototxt_path, caffemodel_path):
    """
    Descargar modelo desde:
    https://github.com/s9xie/hed (prototxt y caffemodel)
    O usar la implementacion de PyImageSearch.
    """
    h, w = img_bgr.shape[:2]
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Preprocesamiento
    mean_pixel = (104.00698793, 116.66876762, 122.67891434)
    blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0,
                                  size=(w, h),
                                  mean=mean_pixel,
                                  swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()

    # Post-procesamiento
    hed = hed[0, 0]  # quitar dimensiones de batch y canal
    hed = (255 * hed).clip(0, 255).astype(np.uint8)
    return hed

# Uso en pipeline de deteccion de borde de pagina:
# 1. hed_map = aplicar_HED(img)
# 2. _, binary_edges = cv2.threshold(hed_map, 80, 255, cv2.THRESH_BINARY)
# 3. lines = cv2.HoughLinesP(binary_edges, ...)
# 4. extraer cuadrilatero desde lines
```

---

### 3.5 Regresion Directa de Esquinas con CNN (Corner Regression)

**Descripcion del algoritmo:**

El enfoque de Dropbox (2019) usa una CNN (DenseNet-121) que directamente predice las coordenadas (x, y) de las 4 esquinas del cuadrilatero como un problema de regresion:

1. **Entrada:** Imagen redimensionada a tamano fijo (ej. 256x256).
2. **Backbone CNN:** DenseNet-121 o MobileNet para eficiencia, sin las capas de clasificacion finales.
3. **Cabeza de regresion:** Capa fully-connected que produce 8 valores: (x1,y1, x2,y2, x3,y3, x4,y4).
4. **Funcion de perdida:** L1 o L2 sobre las coordenadas de las esquinas.
5. **Post-procesamiento:** Orden de las esquinas (top-left, top-right, bottom-right, bottom-left) y reescalado a la dimension original.

**Refinamiento en dos etapas:**

Una variante mas robusta (mencionada en blog Genius Scan 2024) usa dos redes:
- Red 1: detecta las 4 esquinas aproximadas (deteccion gruesa)
- Red 2: refina cada esquina individualmente (deteccion fina por esquina)

**Parametros clave:**
- Arquitectura del backbone (balance entre precision y velocidad)
- Esquema de data augmentation (rotacion, escala, recorte, distorsion optica para simular curvatura)
- Representacion de las esquinas (coordenadas normalizadas o absolutas)

**Fortalezas:**
- End-to-end: no requiere pasos intermedios de deteccion de lineas
- Muy rapido en inferencia (puede ser tiempo real)
- Robusto ante variaciones de iluminacion y fondo si el entrenamiento es representativo
- Maneja bien las oclusiones parciales si hay ejemplos en el entrenamiento

**Debilidades:**
- Requiere muchos ejemplos de entrenamiento anotados con las 4 esquinas
- Puede fallar en esquinas muy ocluidas (ej. lomo que cubre la esquina inferior izquierda)
- La precision de la regresion puede ser insuficiente si se necesita sub-pixel accuracy
- No explota informacion geometrica (no sabe que el resultado debe ser un cuadrilatero convexo)

**Condiciones de buen funcionamiento:**
- Con entrenamiento en imagenes similares al corpus Heraldo
- Para oclusiones parciales (no totales) de las esquinas
- Cuando la velocidad de procesamiento es critica

**Implementacion basica en PyTorch:**

```python
import torch
import torch.nn as nn
import torchvision.models as models

class DetectorEsquinas(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone DenseNet121 pre-entrenado en ImageNet
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        n_features = densenet.classifier.in_features * 7 * 7
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  # 4 esquinas x 2 coordenadas
        )

    def forward(self, x):
        features = self.features(x)
        features = self.pool(features)
        coords = self.regressor(features)
        # coords forma: (batch, 8) -> reshape a (batch, 4, 2)
        return coords.view(-1, 4, 2)

# Entrenamiento:
# loss = nn.MSELoss()(prediccion, ground_truth_corners)
# Augmentation: RandomRotation, RandomResizedCrop, ColorJitter
```

---

### 3.6 Pipeline Hibrido: Segmentacion + RANSAC + Refinamiento Geometrico

**Descripcion del algoritmo:**

Esta tecnica combina multiples enfoques para maximizar robustez:

**Paso 1: Deteccion gruesa de la region de pagina**
- Usar un modelo ligero (ej. MobileNet-SSD fine-tuneado) para detectar la bounding box aproximada de la pagina.
- Alternativa sin entrenamiento: Otsu + morfologia + contorno mas grande.

**Paso 2: Deteccion refinada de bordes dentro de la region**
- Dentro de la bounding box detectada, aplicar HED o Canny para extraer bordes.
- Aplicar LSD para detectar segmentos de linea.
- Clasificar los segmentos por orientacion: horizontales (bordes superior/inferior) y verticales (bordes izquierdo/derecho).

**Paso 3: Ajuste robusto con RANSAC**
- Para cada uno de los 4 bordes, aplicar RANSAC para ajustar la mejor linea a los segmentos detectados.
- RANSAC es robusto a outliers (objetos que sobresalen del margen son outliers para la linea del borde real).

**Paso 4: Interseccion de lineas para encontrar esquinas**
- Calcular las 4 intersecciones de las lineas ajustadas.
- Aplicar restricciones geometricas: el resultado debe ser convexo, el area debe ser razonable.

**Paso 5: Refinamiento sub-pixel**
- Opcionalmente, refinar la posicion de cada esquina buscando el maximo gradiente en un vecindario pequeno.

**Parametros clave RANSAC:**
- `threshold`: distancia maxima de un punto al modelo para ser considerado inlier (ej. 5-10 pixels)
- `max_trials`: numero de iteraciones (ej. 100-500)
- `min_samples`: minimo de puntos para ajustar una linea (2 para lineas)

**Fortalezas:**
- RANSAC es inherentemente robusto a outliers (objetos que sobresalen del margen)
- No requiere entrenamiento
- Funciona bien cuando el borde es al menos 50-60% visible
- La restriccion de clasificacion por orientacion reduce ambiguedad

**Debilidades:**
- El borde inferior (lomo) puede no tener suficientes inliers para RANSAC si esta muy ocluido
- Requiere que los bordes detectados en los pasos anteriores sean de suficiente calidad
- Mas lento que la regresion directa

**Implementacion con scikit-image y OpenCV:**

```python
import cv2
import numpy as np
from skimage.measure import LineModelND, ransac
from skimage.transform import probabilistic_hough_line

def ajustar_linea_ransac(puntos):
    """Ajusta una linea a un conjunto de puntos con RANSAC."""
    if len(puntos) < 10:
        return None
    model, inliers = ransac(puntos, LineModelND,
                             min_samples=2,
                             residual_threshold=5,
                             max_trials=200)
    return model

def detectar_bordes_con_ransac(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detectar segmentos de linea
    lineas = probabilistic_hough_line(edges, threshold=50,
                                      line_length=50, line_gap=10)

    # Separar por orientacion
    horizontales = []
    verticales = []
    for (x0, y0), (x1, y1) in lineas:
        angulo = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        if abs(angulo) < 30:  # horizontal
            # Agregar todos los puntos del segmento
            for t in np.linspace(0, 1, 20):
                horizontales.append([x0 + t*(x1-x0), y0 + t*(y1-y0)])
        elif abs(angulo) > 60:  # vertical
            for t in np.linspace(0, 1, 20):
                verticales.append([x0 + t*(x1-x0), y0 + t*(y1-y0)])

    if len(horizontales) < 10 or len(verticales) < 10:
        return None

    # Separar horizontales en superior e inferior
    h_pts = np.array(horizontales)
    mediana_y = np.median(h_pts[:, 1])
    pts_sup = h_pts[h_pts[:, 1] < mediana_y]
    pts_inf = h_pts[h_pts[:, 1] >= mediana_y]

    # Separar verticales en izquierdo y derecho
    v_pts = np.array(verticales)
    mediana_x = np.median(v_pts[:, 0])
    pts_izq = v_pts[v_pts[:, 0] < mediana_x]
    pts_der = v_pts[v_pts[:, 0] >= mediana_x]

    # Ajustar lineas con RANSAC
    linea_sup = ajustar_linea_ransac(pts_sup)
    linea_inf = ajustar_linea_ransac(pts_inf)
    linea_izq = ajustar_linea_ransac(pts_izq)
    linea_der = ajustar_linea_ransac(pts_der)

    return linea_sup, linea_inf, linea_izq, linea_der
```

---

## 4. Resumen de Recomendaciones para el Corpus Heraldo

### Comparacion rapida de enfoques

| Tecnica | Implementacion | Datos de entrenamiento | Robustez ante oclusiones | Robustez ante pag. adyacentes | Lomo/pliegue | Velocidad |
|---|---|---|---|---|---|---|
| Proyeccion de perfiles | OpenCV/NumPy, trivial | No requiere | Baja | Media | Baja | Muy rapida |
| Canny + Hough (actual) | OpenCV, simple | No requiere | Baja | Baja | Baja | Muy rapida |
| Contraste borde/fondo | OpenCV, moderada | No requiere | Media | Media | Media | Rapida |
| HED como preprocesador | OpenCV DNN | No requiere (pre-entrenado) | Media-Alta | Media | Media | Moderada |
| FCN / U-Net segmentacion | PyTorch/TF | Requiere anotaciones de mascara | Alta | Alta | Media-Alta | Moderada (GPU) |
| Regresion de esquinas CNN | PyTorch/TF | Requiere anotaciones de esquinas | Alta | Alta | Media | Rapida (GPU) |
| RANSAC + segmentos | scikit-image + OpenCV | No requiere | Alta | Media | Media | Moderada |
| dhSegment | Python (implementacion publica) | Requiere anotaciones de mascara | Alta | Alta | Media-Alta | Moderada (GPU) |

### Estrategia recomendada para el pipeline Copista

**Corto plazo (sin requerimientos de datos de entrenamiento):**

1. Reemplazar Canny + Hough por **HED** como primer paso de deteccion de bordes. HED produce bordes mucho mas limpios y es menos sensible a los parametros. Implementable directamente con `cv2.dnn`.

2. Reemplazar la transformada de Hough estandar por **RANSAC sobre segmentos LSD** para ajustar las lineas de borde. RANSAC maneja nativamente los objetos que sobresalen del margen (son outliers).

3. Agregar el **criterio de contraste** para seleccionar entre multiples hipotesis de cuadrilatero, descartando las que tengan bajo contraste interior/exterior.

**Mediano plazo (requiere anotacion de ~200-500 imagenes):**

4. Entrenar un modelo de **segmentacion semantica** (dhSegment o U-Net ligero) usando las imagenes del corpus Heraldo anotadas con mascaras de pagina. Este modelo reemplaza los pasos 1-3 y es mucho mas robusto.

5. Alternativamente, entrenar un **modelo de regresion de esquinas** (DenseNet ligero o MobileNet) que prediga directamente las 4 esquinas. Requiere anotaciones de las 4 esquinas, que son mas faciles de generar que mascaras completas.

**Para el problema especifico del lomo/pliegue:**

El borde inferior del lomo es estructuralmente diferente: no hay un borde limpio sino una transicion gradual con curvatura. Las tecnicas recomendadas son:
- **RANSAC**: es robusto si hay suficientes pixels del borde visible
- **Modelo aprendido**: si se incluyen ejemplos de lomos en el entrenamiento
- Como fallback: usar una linea horizontal estimada a partir de los bordes laterales (si conocemos la geometria de la pagina, podemos extrapolar)

---

## 5. Datasets de Benchmark Relevantes

| Dataset | Descripcion | URL |
|---|---|---|
| CBDAR 2007 (DFKI-I) | Imagenes de documentos capturados con camara, usado en Stamatopoulos et al. y Bukhari et al. | Buscar en sitio de CBDAR workshop |
| MIDV-500 | 500 clips de video de 50 tipos de documentos de identidad, con ground truth de esquinas | https://arxiv.org/pdf/1807.05786 |
| SmartDoc 2015 | Imagenes de documentos desde smartphone para evaluacion de deteccion y segmentacion | Buscar en sitio de ICDAR |
| DocUNet benchmark | Imagenes de documentos doblados/curvados con ground truth de dewarping | Disponible con el codigo de DocUNet |
| UW-III | Dataset clasico de documentos escaneados para OCR y analisis de layout | Universidad de Washington |
| READ-BAD / cBAD | Documentos manuscritos historicos (siglos XVII-XX) para analisis de baseline | Proyecto READ |

---

## 6. Referencias en Formato Bibliografico

1. Shafait, F., van Beusekom, J., Keysers, D., & Breuel, T.M. (2010). Document cleanup using page frame detection. *International Journal on Document Analysis and Recognition (IJDAR)*, 13(1), 43-52. https://doi.org/10.1007/s10032-008-0071-7

2. Shamqoli, A., & Khosravi, H. (2013/2014). Border detection of document images scanned from large books. *Iranian Conference on Machine Vision and Image Processing (MVIP)*. IEEE. https://ieeexplore.ieee.org/document/6779955

3. Stamatopoulos, N., Gatos, B., & Kesidis, A.L. (2007). Automatic Borders Detection of Camera Document Images. *Proceedings of the 2nd International Workshop on Camera-Based Document Analysis and Recognition (CBDAR 2007)*. https://users.iit.demokritos.gr/~bgat/CBDAR_BORDERS.pdf

4. Bukhari, S.S., Shafait, F., & Breuel, T.M. (2012). Border Noise Removal of Camera-Captured Document Images Using Page Frame Detection. In: *Camera-Based Document Analysis and Recognition. CBDAR 2011*. Lecture Notes in Computer Science, vol. 7139. Springer. https://doi.org/10.1007/978-3-642-29364-1_10

5. [Autores, U. Kaiserslautern]. (2018). A Robust Page Frame Detection Method for Complex Historical Document Images. https://www.researchgate.net/publication/329760609

6. Tensmeyer, C., Davis, B.L., Wigington, C., Lee, I., & Barrett, B. (2017). PageNet: Page Boundary Extraction in Historical Handwritten Documents. *Proceedings of the 4th International Workshop on Historical Document Imaging and Processing (HIP 2017)*. https://arxiv.org/abs/1709.01618

7. Oliveira, S.A., Seguin, B., & Kaplan, F. (2018). dhSegment: A generic deep-learning approach for document segmentation. *2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR)*, pp. 7-12. IEEE. https://arxiv.org/abs/1804.10371

8. Wick, C., & Puppe, F. (2018). Fully Convolutional Neural Networks for Page Segmentation of Historical Document Images. *2018 13th IAPR International Workshop on Document Analysis Systems (DAS)*. IEEE. https://arxiv.org/abs/1711.07695

9. Skoryukina, N., et al. (2020). Approach for Document Detection by Contours and Contrasts. *arXiv:2008.02615*. https://arxiv.org/abs/2008.02615

10. Xie, S., & Tu, Z. (2015). Holistically-Nested Edge Detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV 2015)*, pp. 1395-1403. https://arxiv.org/abs/1504.06375

11. Ma, K., Shu, Z., et al. (2018). DocUNet: Document Image Unwarping via a Stacked U-Net. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018)*. https://ieeexplore.ieee.org/document/8578592/

12. Das, S., Ma, K., Shu, Z., Samaras, D., & Shilkrot, R. (2019). DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks. *IEEE/CVF International Conference on Computer Vision (ICCV 2019)*, pp. 131-140. https://openaccess.thecvf.com/content_ICCV_2019/html/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_Networks_ICCV_2019_paper.html

13. Binmakhashen, G.M., & Mahmoud, S.A. (2019). Document Layout Analysis: A Comprehensive Survey. *ACM Computing Surveys*, 52(6), Article 103. https://doi.org/10.1145/3355610

14. Dropbox ML Team. (2019-2020). Fast and Accurate Document Detection for Scanning. *Dropbox Engineering Blog*. https://dropbox.tech/machine-learning/fast-and-accurate-document-detection-for-scanning

15. Simistira Liwicki, F., et al. (2021). Deep Learning for Historical Document Analysis and Recognition -- A Survey. *Journal of Imaging*, 6(10), 110. https://doi.org/10.3390/jimaging6100110

16. Feng, H., et al. (2021/2025). DocScanner: Robust Document Image Rectification with Progressive Learning. *International Journal of Computer Vision (IJCV)*. https://arxiv.org/abs/2110.14968

17. [Autores]. (2024). Handheld Video Document Scanning: A Robust On-Device Model for Multi-Page Document Scanning. *arXiv:2411.00576*. https://arxiv.org/abs/2411.00576

18. Feng, H., et al. (2021). DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction. *ACM MM 2021*. https://arxiv.org/abs/2110.12942

19. [Autores]. (2021). Page Layout Analysis System for Unconstrained Historic Documents. *ICDAR 2021*. https://doi.org/10.1007/978-3-030-86331-9_32

20. [Autores]. (2009-2012). Page frame detection for double page document images. ResearchGate. https://www.researchgate.net/publication/220933023

---

*Documento generado en Marzo 2026 para el proyecto Copista-Pipeline. Revision bibliografica orientada a la implementacion practica en Python con OpenCV.*
