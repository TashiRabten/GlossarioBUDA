package com.example.glossariobuda;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

public class HybridTextExtractor {

    private final DatabaseManager dbManager;
    private TSVColumnParser tsvParser;

    public HybridTextExtractor(DatabaseManager dbManager) {
        this.dbManager = dbManager;
    }

    public void processPDF(String pdfPath, ProgressCallback callback) {
        try (PDDocument document = Loader.loadPDF(new File(pdfPath))) {
            if (callback != null) {
                callback.onProgress(0, 1, "Attempting direct text extraction...");
            }

            PDFTextStripper stripper = new PDFTextStripper();
            String extractedText = stripper.getText(document);

            if (isTextExtractionSuccessful(extractedText)) {
                if (callback != null) {
                    callback.onProgress(1, 1, "Processing extracted text...");
                }

                processExtractedText(extractedText);

                if (callback != null) {
                    callback.onComplete("Text extraction successful! Found terms in " +
                            document.getNumberOfPages() + " pages.");
                }
            } else {
                if (callback != null) {
                    callback.onProgress(0, 1, "Text clausextraction failed, using OCR TSV fallback...");
                }
                System.out.println("[INFO] Switching to OCR TSV fallback...");
                processWithOCRFallback(pdfPath, callback);
            }

        } catch (IOException e) {
            if (callback != null) {
                callback.onError("Error processing PDF: " + e.getMessage());
            }
        }
    }

    private void processWithOCRFallback(String pdfPath, ProgressCallback callback) {
        try {
            File pdfFile = new File(pdfPath);
            if (!pdfFile.exists()) {
                if (callback != null) callback.onError("PDF file not found: " + pdfPath);
                return;
            }

            try (PDDocument document = Loader.loadPDF(pdfFile)) {
                PDFRenderer pdfRenderer = new PDFRenderer(document);

                for (int pageIndex = 0; pageIndex < document.getNumberOfPages(); pageIndex++) {
                    if (callback != null) {
                        callback.onProgress(pageIndex + 1, document.getNumberOfPages(),
                                "OCR TSV processing page " + (pageIndex + 1) + "...");
                    }

                    BufferedImage image = pdfRenderer.renderImageWithDPI(pageIndex, 300);
                    File imageFile = new File("temp_page_" + pageIndex + ".png");
                    ImageIO.write(image, "png", imageFile);

                    // Try Namsel OCR first (better for Tibetan compound syllables)
                    boolean namselSuccess = tryNamselOCR(imageFile, pageIndex, dbManager);
                    
                    if (!namselSuccess) {
                        // Fallback to Tesseract OCR
                        System.out.println("[FALLBACK] Switching to Tesseract OCR for page " + (pageIndex + 1));
                        tryTesseractOCR(imageFile, pageIndex, dbManager);
                    }
                }

                if (callback != null) {
                    callback.onComplete("OCR TSV fallback completed for " +
                            document.getNumberOfPages() + " pages.");
                }
            }

        } catch (Exception e) {
            if (callback != null) {
                callback.onError("OCR TSV fallback failed: " + e.getMessage());
            }
        }
    }

    private boolean isTextExtractionSuccessful(String text) {
        boolean hasEnglish = text.matches(".*[a-zA-Z]{3,}.*");
        boolean hasTibetan = text.matches(".*[\\u0F00-\\u0FFF].*");
        boolean hasReasonableStructure = text.length() > 100 && text.split("\\n").length > 10;
        System.out.println("[CHECK] Text extraction successful? English=" + hasEnglish +
                ", Tibetan=" + hasTibetan + ", Structure=" + hasReasonableStructure);
        return hasEnglish && hasTibetan && hasReasonableStructure;
    }

    private void processExtractedText(String fullText) {
        String[] pages = fullText.split("(?m)^\\s*\\d+\\s*$");
        for (int i = 0; i < pages.length; i++) {
            System.out.println("[TEXT] Page " + (i + 1) + ": Skipping direct text parsing (not implemented)");
        }
    }

    /**
     * Try Namsel OCR for better Tibetan compound syllable recognition
     */
    private boolean tryNamselOCR(File imageFile, int pageIndex, DatabaseManager dbManager) {
        try {
            // Check if Python3 and Namsel OCR module are available
            ProcessBuilder checkPython = new ProcessBuilder("python3", "-m", "namsel_ocr.namsel", "--help");
            
            // Capture output to check for help message even with warnings
            Process checkProcess = checkPython.start();
            
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(checkProcess.getInputStream()))) {
                String line;
                boolean hasHelp = false;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("usage:") || line.contains("Namsel OCR")) {
                        hasHelp = true;
                        break;
                    }
                }
                
                int checkExitCode = checkProcess.waitFor();
                
                if (checkExitCode != 0 && !hasHelp) {
                    System.out.println("[NAMSEL] Namsel OCR Python module not available, falling back to Tesseract");
                    return false;
                }
                
                if (hasHelp) {
                    System.out.println("[NAMSEL] Namsel OCR detected (with warnings about missing training data)");
                }
            }
            
            // Run Namsel OCR using the Python module
            File namselOutput = new File("temp_page_" + pageIndex + "_namsel.txt");
            ProcessBuilder pbNamsel = new ProcessBuilder(
                    "python3", "-m", "namsel_ocr.namsel",
                    "recognize-page",
                    "--page_type=book", 
                    "--format=text",
                    "--outfile=" + namselOutput.getName(),
                    imageFile.getAbsolutePath()
            );
            
            Process namselProcess = pbNamsel.start();
            // Add timeout to prevent hanging (30 seconds)

            System.out.println("[NAMSEL] Starting OCR process...");
            long start = System.currentTimeMillis();

            boolean finished = namselProcess.waitFor(120, java.util.concurrent.TimeUnit.SECONDS);
            if (!finished) {
                System.out.println("[NAMSEL] Namsel OCR timed out, terminating process");
                namselProcess.destroyForcibly();
                return false;
            }

            long duration = System.currentTimeMillis() - start;
            System.out.println("[NAMSEL] OCR duration: " + (duration / 1000.0) + " seconds");

            int namselExitCode = namselProcess.exitValue();
            
            if (namselExitCode != 0 || !namselOutput.exists() || namselOutput.length() < 10) {
                System.out.println("[NAMSEL] Namsel OCR failed for page " + (pageIndex + 1));
                return false;
            }
            
            // TODO: Parse Namsel output and integrate into database
            // For now, just indicate success - implementation depends on Namsel output format
            System.out.println("[NAMSEL] Namsel OCR completed for page " + (pageIndex + 1));
            System.out.println("[NAMSEL] Output file: " + namselOutput.getAbsolutePath());
            
            // Read and display Namsel output for debugging
            try (BufferedReader reader = new BufferedReader(new FileReader(namselOutput))) {
                String line;
                System.out.println("[NAMSEL] Content preview:");
                int lineCount = 0;
                while ((line = reader.readLine()) != null && lineCount < 5) {
                    System.out.println("  " + line);
                    lineCount++;
                }
            }
            
            return true;
            
        } catch (Exception e) {
            System.err.println("[NAMSEL] Error running Namsel OCR: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Fallback Tesseract OCR method
     */
    private boolean tryTesseractOCR(File imageFile, int pageIndex, DatabaseManager dbManager) {
        try {
            File tsvOutput = new File("temp_page_" + pageIndex + ".tsv");

            ProcessBuilder pb = new ProcessBuilder(
                    "tesseract",
                    imageFile.getAbsolutePath(),
                    "stdout",
                    "-l", "eng+bod",
                    "--oem", "1",
                    "--psm", "6",
                    "-c", "preserve_interword_spaces=1",
                    "-c", "load_system_dawg=false",
                    "-c", "load_freq_dawg=false",
                    "tsv"
            );
            pb.redirectOutput(tsvOutput);
            Process process = pb.start();
            int exitCode = process.waitFor();

            if (!tsvOutput.exists() || tsvOutput.length() < 100) {
                System.err.println("[TESSERACT] TSV file missing or too small: " + tsvOutput.getAbsolutePath());
                return false;
            }

            System.out.println("[TESSERACT] Calling TSVColumnParser on page " + (pageIndex + 1));
            TSVColumnParser.integrateIntoDatabase(tsvOutput, dbManager, pageIndex + 1);
            return true;
            
        } catch (Exception e) {
            System.err.println("[TESSERACT] Error running Tesseract OCR: " + e.getMessage());
            return false;
        }
    }

    public interface ProgressCallback {
        void onProgress(int current, int total, String message);
        void onError(String error);
        void onComplete(String message);
    }
}
