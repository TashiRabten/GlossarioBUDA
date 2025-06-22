@echo off
echo Building GlossarioBUDA Installer...

rem Step 1: Clean and compile
call mvn clean compile

rem Step 2: Create runtime image with jlink
call mvn jlink:jlink

rem Step 3: Copy tessdata to runtime image (if exists)
if exist "src\main\resources\tessdata" (
    xcopy /s /y "src\main\resources\tessdata" "target\GlossarioBUDA\tessdata\"
)

rem Step 4: Create installer with jpackage
jpackage ^
    --type exe ^
    --name "GlossarioBUDA" ^
    --app-version "1.0" ^
    --vendor "Associação Buddha-Dharma" ^
    --description "Sistema de Terminologia Budista" ^
    --runtime-image "target\GlossarioBUDA" ^
    --module com.example.glossariobuda/com.example.glossariobuda.GlossarioBUDAMain ^
    --dest installer ^
    --win-console ^
    --win-dir-chooser ^
    --win-menu ^
    --win-shortcut ^
    --file-associations file-associations.properties

echo Build complete! Installer created in 'installer' directory.
pause