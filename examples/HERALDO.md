### Procesar prueba Heraldo

Descargar set de imagenes de 100 imagenes random (50 derechas y 50 izquierdas):

[Heraldo_seleccion_random.zip](https://files.bibliohack.org/Heraldo_seleccion_random.zip)

Crear la carpeta `__data` en el repositorio y las carpetas para el procesamiento del proyecto

    cd Copista-Pipeline
    mkdir __data
    mkdir __data/heraldo_raw
    mkdir __data/heraldo_processed
    mkdir __data/heraldo_PDFs
    
Descomprimir el zip y copiar las carpetas `Derecha` e `Izquierda` a `__data/heraldo_raw/`

Para establecer el entorno de procesamiento instalar los siguiuentes paquetes:

    # Instalar dependencias  
    sudo apt install python3-venv tesseract-ocr tesseract-ocr-spa
    # Crear Python Virtual Environment en el repositorio
    cd Copista-Pipeline
    python3 -m venv venv 
    # Activar virtual environment
    source venv/bin/activate 
    # En la terminal aparecerá el (venv) $
    # Instalar dependencias en el virtual environment
    pip install opencv-python numpy tqdm pytesseract pillow hocr-tools reportlab

Para inciar el proceso ejecutar los comandos desde la carpeta raiz del proyecto:

    python3 src/batch_processor.py --pipeline ./examples/Heraldo_Derecha/
    python3 src/batch_processor.py --pipeline ./examples/Heraldo_OCR_Derecha/
    python3 src/batch_processor.py --pipeline ./examples/Heraldo_Izquierda/
    python3 src/batch_processor.py --pipeline ./examples/Heraldo_OCR_Izquierda/

Para unir los pdfs

    cd __data/heraldo_PDFs/Derecha
    pdfunite *.pdf paginas_derechas_random.pdf
    cd __data/heraldo_PDFs/Izquierda
    pdfunite *.pdf paginas_izquierdas_random.pdf

