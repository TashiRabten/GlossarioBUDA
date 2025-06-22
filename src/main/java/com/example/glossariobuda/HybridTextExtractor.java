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
                // Fallback to OCR
                if (callback != null) {
                    callback.onProgress(0, 1, "Text extraction failed, falling back to OCR...");
                }

                // Convert callback interface
                OCRProcessor.ProgressCallback ocrCallback = new OCRProcessor.ProgressCallback() {
                    @Override
                    public void onProgress(int current, int total, String message) {
                        if (callback != null) callback.onProgress(current, total, message);
                    }
                    
                    @Override
                    public void onError(String error) {
                        if (callback != null) callback.onError(error);
                    }
                    
                    @Override
                    public void onComplete(String message) {
                        if (callback != null) callback.onComplete(message);
                    }
                };
                
                ocrFallback.processPDF(pdfPath, ocrCallback);
            }

        } catch (IOException e) {
            if (callback != null) {
                callback.onError("Error processing PDF: " + e.getMessage());
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
            String context = "Page " + pageNumber + " (Direct extraction)";
            if (term.subject != null && !term.subject.isEmpty()) {
                context += " | Subject: " + term.subject;
            }

            dbManager.addTerm(
                    term.englishTerm,
                    "English",
                    term.tibetanTerm,
                    "Tibetan",
                    context,
                    "TextExtractor-Bot",
                    "Requires manual review"
            );
        }
    }

    private List<TermPair> extractTermsFromText(String text) {
        List<TermPair> terms = new ArrayList<>();

        // Pattern optimized for direct text extraction (cleaner than OCR)
        Pattern termPattern = Pattern.compile(
                "([a-zA-Z][a-zA-Z\\s-']+?)\\s+" +              // English term
                        "([a-z]+(?:[.,]\\s*[a-z]+)*)\\s+" +            // Subject codes
                        "([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།།]+)", // Tibetan text
                Pattern.MULTILINE
        );

        Matcher matcher = termPattern.matcher(text);
        while (matcher.find()) {
            String englishTerm = matcher.group(1).trim().toLowerCase();
            String subjects = matcher.group(2);
            String tibetanTerm = matcher.group(3).trim();

            // Clean up and validate
            englishTerm = englishTerm.replaceAll("\\s+", " ");
            tibetanTerm = tibetanTerm.replaceAll("\\s+", " ");

            if (isValidTerm(englishTerm, tibetanTerm)) {
                terms.add(new TermPair(englishTerm, tibetanTerm, subjects));
            }
        }

        return terms;
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

        TermPair(String englishTerm, String tibetanTerm, String subject) {
            this.englishTerm = englishTerm;
            this.tibetanTerm = tibetanTerm;
            this.subject = subject;
        }
    }
}