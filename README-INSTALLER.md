# Building GlossarioBUDA Installer

## Prerequisites
1. **Java 17+** with JDK (for jpackage)
2. **Maven** installed
3. **Tesseract language files** downloaded

## Setup Steps

### 1. Download Tesseract Language Files
Download these files to `src/main/resources/tessdata/`:

- **bod.traineddata** (Tibetan): https://github.com/tesseract-ocr/tessdata_best/blob/main/bod.traineddata
- **eng.traineddata** (English): https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata

### 2. Bundle Tesseract Executables (Optional)
For complete self-contained app, download Tesseract Windows binaries:
- Place `tesseract.exe` in `src/main/resources/tesseract/`
- Include required DLLs

### 3. Build Installer
Run in PowerShell/CMD:
```bash
build-installer.bat
```

## Output
- **Installer**: `installer/GlossarioBUDA-1.0.exe`
- **Portable**: `target/GlossarioBUDA/` (runtime image)

## Features Bundled
✅ Java Runtime (no JRE needed on target machine)
✅ JavaFX libraries
✅ SQLite database
✅ PDF processing (PDFBox)
✅ Tesseract language files
✅ All dependencies

## Installation
The generated .exe installer will:
- Install to Program Files
- Create desktop shortcut
- Add to Start Menu
- Associate with PDF files
- Include uninstaller

## Troubleshooting
- Ensure JDK 17+ is installed (not just JRE)
- Run from Command Prompt with Administrator privileges
- Check that Maven is in PATH