package com.example.glossariobuda;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Alternative text extraction approach that tries direct PDF text extraction first,
 * then falls back to OCR if needed.
 */
public class HybridTextExtractor {
    private final DatabaseManager dbManager;
    private final OCRProcessor ocrFallback;

    public HybridTextExtractor(DatabaseManager dbManager) {
        this.dbManager = dbManager;
        this.ocrFallback = new OCRProcessor(dbManager);
    }

    public void processPDF(String pdfPath, ProgressCallback callback) {
        try (PDDocument document = Loader.loadPDF(new File(pdfPath))) {

            // First, try direct text extraction
            if (callback != null) {
                callback.onProgress(0, 1, "Attempting direct text extraction...");
            }

            PDFTextStripper stripper = new PDFTextStripper();
            String extractedText = stripper.getText(document);

            // Check if the extracted text contains meaningful content
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
                // Fallback to OCR with column detection
                if (callback != null) {
                    callback.onProgress(0, 1, "Text extraction failed, using OCR with column detection...");
                }

                processWithOCRAndColumnDetection(pdfPath, callback);
            }

        } catch (IOException e) {
            if (callback != null) {
                callback.onError("Error processing PDF: " + e.getMessage());
            }
        }
    }
    
    private void processWithOCRAndColumnDetection(String pdfPath, ProgressCallback callback) {
        // Use OCR to extract text, then apply our column detection
        try {
            System.out.println("[HYBRID] Starting OCR processing with column detection...");
            
            // Get OCR text using the existing OCR processor
            java.io.File pdfFile = new java.io.File(pdfPath);
            if (!pdfFile.exists()) {
                if (callback != null) callback.onError("PDF file not found: " + pdfPath);
                return;
            }
            
            // Use PDFBox to convert to images and OCR each page
            try (org.apache.pdfbox.pdmodel.PDDocument document = org.apache.pdfbox.Loader.loadPDF(pdfFile)) {
                org.apache.pdfbox.rendering.PDFRenderer pdfRenderer = new org.apache.pdfbox.rendering.PDFRenderer(document);
                
                for (int pageIndex = 0; pageIndex < document.getNumberOfPages(); pageIndex++) {
                    if (callback != null) {
                        callback.onProgress(pageIndex + 1, document.getNumberOfPages(), 
                            "Processing page " + (pageIndex + 1) + " with OCR and column detection...");
                    }
                    
                    // Convert page to image and OCR it
                    java.awt.image.BufferedImage image = pdfRenderer.renderImageWithDPI(pageIndex, 300);
                    
                    // OCR the image
                    net.sourceforge.tess4j.Tesseract tesseract = new net.sourceforge.tess4j.Tesseract();
                    tesseract.setDatapath("C:/Program Files/Tesseract-OCR/tessdata");
                    tesseract.setLanguage("eng+bod");
                    
                    String ocrText = tesseract.doOCR(image);
                    
                    System.out.println("[HYBRID] OCR text for page " + (pageIndex + 1) + ": " + 
                        ocrText.substring(0, Math.min(100, ocrText.length())) + "...");
                    
                    // Apply column detection to OCR text
                    processPageText(ocrText, pageIndex + 1);
                }
                
                if (callback != null) {
                    callback.onComplete("OCR with column detection completed for " + 
                        document.getNumberOfPages() + " pages.");
                }
                
            }
            
        } catch (Exception e) {
            System.err.println("[HYBRID] OCR processing failed: " + e.getMessage());
            e.printStackTrace();
            if (callback != null) {
                callback.onError("OCR processing failed: " + e.getMessage());
            }
        }
    }

    private boolean isTextExtractionSuccessful(String text) {
        // Check if the text contains both English and Tibetan characters
        boolean hasEnglish = text.matches(".*[a-zA-Z]{3,}.*");
        boolean hasTibetan = text.matches(".*[\\u0F00-\\u0FFF].*");

        // Check if it has reasonable structure (not just garbled text)
        boolean hasReasonableStructure = text.length() > 100 &&
                text.split("\\n").length > 10;

        return hasEnglish && hasTibetan && hasReasonableStructure;
    }

    private void processExtractedText(String fullText) {
        // Split by pages if possible (look for page breaks or numbers)
        String[] pages = splitIntoPages(fullText);

        for (int i = 0; i < pages.length; i++) {
            processPageText(pages[i], i + 1);
        }
    }

    private String[] splitIntoPages(String fullText) {
        // Try to split by page numbers or other indicators
        // This is a simple approach - you might need to adjust based on your PDF structure

        List<String> pages = new ArrayList<>();
        String[] lines = fullText.split("\\n");
        StringBuilder currentPage = new StringBuilder();

        for (String line : lines) {
            // Check if this line indicates a new page
            if (isPageBreak(line)) {
                if (currentPage.length() > 0) {
                    pages.add(currentPage.toString());
                    currentPage = new StringBuilder();
                }
            } else {
                currentPage.append(line).append("\n");
            }
        }

        // Add the last page
        if (currentPage.length() > 0) {
            pages.add(currentPage.toString());
        }

        return pages.toArray(new String[0]);
    }

    private boolean isPageBreak(String line) {
        // Detect page breaks - adjust based on your PDF format
        return line.matches("^\\s*\\d+\\s*$") || // Just a page number
                line.matches(".*absolute pressure.*\\d+") || // Header format from your PDF
                line.length() < 5 && line.matches("\\s*");
    }

    private void processPageText(String pageText, int pageNumber) {
        List<TermPair> terms = extractTermsFromText(pageText);

        for (TermPair term : terms) {
            // Reconstruct single line output with all columns properly formatted
            String reconstructedLine = reconstructSingleLineOutput(term);
            
            String context = "Page " + pageNumber + " (Hybrid Column Detection)";
            if (term.subject != null && !term.subject.isEmpty()) {
                context += " | Subject: " + term.subject;
            }
            
            // Add definition to context if available
            if (term.tibetanDefinition != null && !term.tibetanDefinition.trim().isEmpty()) {
                context += " | Definition: " + term.tibetanDefinition;
            }

            // Store the Tibetan term and definition properly
            dbManager.addTerm(
                    term.englishTerm,
                    "English",
                    term.tibetanTerm,  // Only the term, not the definition
                    "Tibetan",
                    context,
                    "HybridColumnDetector-Bot",
                    "Requires manual review"
            );
            
            // Output the reconstructed single line for verification
            System.out.println("[RECONSTRUCTED] " + reconstructedLine);
            System.out.println("[STORED] " + term.englishTerm + " -> Term: '" + 
                term.tibetanTerm + "' | Def: '" + 
                (term.tibetanDefinition.length() > 30 ? term.tibetanDefinition.substring(0, 30) + "..." : term.tibetanDefinition) + "'");
        }
    }
    
    /**
     * Reconstruct single line output with proper column alignment
     * Format: [English Term] | [Subject] | [Tibetan Term] | [Tibetan Definition]
     */
    private String reconstructSingleLineOutput(TermPair term) {
        // Ensure consistent formatting with proper spacing
        String english = term.englishTerm != null ? term.englishTerm.trim() : "";
        String subject = term.subject != null ? term.subject.trim() : "";
        String tibetanTerm = term.tibetanTerm != null ? term.tibetanTerm.trim() : "";
        String tibetanDef = term.tibetanDefinition != null ? term.tibetanDefinition.trim() : "";
        
        // Format with column separators for clear visualization
        StringBuilder line = new StringBuilder();
        
        // Column 1: English (pad to ~20 chars for alignment)
        line.append(String.format("%-20s", english.length() > 20 ? english.substring(0, 17) + "..." : english));
        line.append(" | ");
        
        // Column 2: Subject (pad to ~8 chars)
        line.append(String.format("%-8s", subject.length() > 8 ? subject.substring(0, 8) : subject));
        line.append(" | ");
        
        // Column 3: Tibetan Term (pad to ~25 chars)
        String displayTerm = tibetanTerm.length() > 25 ? tibetanTerm.substring(0, 22) + "..." : tibetanTerm;
        line.append(String.format("%-25s", displayTerm));
        line.append(" | ");
        
        // Column 4: Tibetan Definition (truncate if too long)
        String displayDef = tibetanDef.length() > 50 ? tibetanDef.substring(0, 47) + "..." : tibetanDef;
        line.append(displayDef);
        
        return line.toString();
    }

    private List<TermPair> extractTermsFromText(String text) {
        List<TermPair> terms = new ArrayList<>();

        System.out.println("[HYBRID] Using ColumnDetector for text extraction");
        
        // Use the sophisticated ColumnDetector instead of simple regex
        List<ColumnDetector.ColumnizedTerm> columnizedTerms = ColumnDetector.detectColumns(text);
        
        for (ColumnDetector.ColumnizedTerm columnTerm : columnizedTerms) {
            if (isValidColumnTerm(columnTerm)) {
                terms.add(new TermPair(
                    columnTerm.englishTerm, 
                    columnTerm.tibetanTerm,  // Store only the term, not combined
                    columnTerm.subject,
                    columnTerm.tibetanDefinition  // Store definition separately
                ));
            }
        }

        System.out.println("[HYBRID] ColumnDetector found " + terms.size() + " valid terms");
        return terms;
    }

    private boolean isValidColumnTerm(ColumnDetector.ColumnizedTerm columnTerm) {
        if (columnTerm == null) return false;
        
        String englishTerm = columnTerm.englishTerm;
        String tibetanTerm = columnTerm.tibetanTerm;
        
        if (englishTerm == null || tibetanTerm == null) return false;
        if (englishTerm.trim().length() < 2 || tibetanTerm.trim().length() < 2) return false;
        if (englishTerm.length() > 100) return false; // Too long, probably an error

        // Should contain actual Tibetan characters
        if (!tibetanTerm.matches(".*[\\u0F00-\\u0FFF].*")) return false;

        // Should be reasonable English words
        if (!englishTerm.matches("[a-zA-Z][a-zA-Z\\s'-]*")) return false;

        return true;
    }
    
    private String combineTibetanFields(String tibetanTerm, String tibetanDefinition) {
        StringBuilder combined = new StringBuilder();
        
        if (tibetanTerm != null && !tibetanTerm.trim().isEmpty()) {
            combined.append(tibetanTerm.trim());
        }
        
        if (tibetanDefinition != null && !tibetanDefinition.trim().isEmpty()) {
            if (combined.length() > 0) {
                combined.append(" ");
            }
            combined.append(tibetanDefinition.trim());
        }
        
        return combined.toString();
    }

    private boolean isValidTerm(String englishTerm, String tibetanTerm) {
        if (englishTerm == null || tibetanTerm == null) return false;
        if (englishTerm.length() < 2 || tibetanTerm.length() < 2) return false;
        if (englishTerm.length() > 100) return false; // Too long, probably an error

        // Should contain actual Tibetan characters
        if (!tibetanTerm.matches(".*[\\u0F00-\\u0FFF].*")) return false;

        // Should be reasonable English words
        if (!englishTerm.matches("[a-zA-Z][a-zA-Z\\s'-]*")) return false;

        return true;
    }

    public interface ProgressCallback {
        void onProgress(int current, int total, String message);
        void onError(String error);
        void onComplete(String message);
    }

    private static class TermPair {
        final String englishTerm;
        final String tibetanTerm;
        final String subject;
        final String tibetanDefinition;

        TermPair(String englishTerm, String tibetanTerm, String subject, String tibetanDefinition) {
            this.englishTerm = englishTerm;
            this.tibetanTerm = tibetanTerm;
            this.subject = subject;
            this.tibetanDefinition = tibetanDefinition != null ? tibetanDefinition : "";
        }
    }
}