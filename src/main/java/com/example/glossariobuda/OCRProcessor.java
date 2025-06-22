package com.example.glossariobuda;

import net.sourceforge.tess4j.Tesseract;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class OCRProcessor {
    private final Tesseract tesseract;
    private final DatabaseManager dbManager;
    private static final boolean DEBUG = true;

    public OCRProcessor(DatabaseManager dbManager) {
        this.dbManager = dbManager;
        this.tesseract = new Tesseract();
        setupTesseract();
    }

    private void debug(String message) {
        if (DEBUG) {
            System.out.println("[OCR DEBUG] " + message);
        }
    }

    private void setupTesseract() {
        debug("Setting up Tesseract...");

        String tessDataPath = findTessDataPath();
        if (tessDataPath != null) {
            tesseract.setDatapath(tessDataPath);
            debug("Tesseract data path: " + tessDataPath);
        }

        try {
            tesseract.setLanguage("eng+bod");
            debug("Language set to: eng+bod");
        } catch (Exception e) {
            debug("Failed to set eng+bod, trying eng only: " + e.getMessage());
            try {
                tesseract.setLanguage("eng");
                debug("Language set to: eng");
            } catch (Exception ex) {
                debug("Using default language");
            }
        }

        tesseract.setOcrEngineMode(1);
        tesseract.setPageSegMode(6);
        tesseract.setVariable("user_defined_dpi", "400");
        tesseract.setVariable("preserve_interword_spaces", "1");

        // Remove character whitelist to test Tibetan recognition
        // tesseract.setVariable("tessedit_char_whitelist", "...");

        debug("Tesseract configured");
    }

    private String findTessDataPath() {
        String[] paths = {
            System.getProperty("user.dir") + "/tessdata",
            "C:\\Program Files\\Tesseract-OCR\\tessdata",
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/local/share/tessdata",
            "/opt/homebrew/share/tessdata"
        };

        for (String path : paths) {
            if (new File(path).exists()) {
                debug("Found tessdata at: " + path);
                return path;
            }
        }

        debug("Warning: No tessdata path found");
        return null;
    }

    public List<String> processPDF(String pdfPath, ProgressCallback callback) {
        debug("=== Starting PDF Processing ===");
        debug("PDF Path: " + pdfPath);

        List<String> extractedText = new ArrayList<>();

        try (PDDocument document = Loader.loadPDF(new File(pdfPath))) {
            debug("PDF loaded. Pages: " + document.getNumberOfPages());

            // Phase 1: Try direct text extraction
            if (callback != null) {
                callback.onProgress(0, 1, "Attempting direct text extraction...");
            }

            PDFTextStripper stripper = new PDFTextStripper();
            String directText = stripper.getText(document);
            debug("Direct text extraction length: " + directText.length());

            if (hasValidGlossaryStructure(directText)) {
                debug("Direct text extraction successful - processing...");
                if (callback != null) {
                    callback.onProgress(1, 1, "Processing direct text extraction...");
                }

                processExtractedText(directText);

                if (callback != null) {
                    callback.onComplete("Direct text extraction successful!");
                }
                extractedText.add(directText);
                return extractedText;
            }

            debug("Direct text extraction failed - falling back to OCR");

            // Phase 2: OCR fallback
            PDFRenderer renderer = new PDFRenderer(document);
            int totalPages = document.getNumberOfPages();

            if (callback != null) {
                callback.onProgress(0, totalPages, "Text extraction failed, using OCR...");
            }

            // Process more pages to reach main glossary (starts around page 25+)
            int pagesToProcess = Math.min(totalPages, 35);
            for (int page = 0; page < pagesToProcess; page++) {
                debug("=== Processing page " + (page + 1) + " ===");

                if (callback != null) {
                    callback.onProgress(page + 1, totalPages, "OCR page " + (page + 1));
                }

                try {
                    BufferedImage image = renderer.renderImageWithDPI(page, 400, ImageType.RGB);
                    String pageText = tesseract.doOCR(image);
                    extractedText.add(pageText);

                    debug("Raw OCR text length: " + pageText.length());
                    if (pageText.length() > 0) {
                        String preview = pageText.substring(0, Math.min(200, pageText.length()));
                        debug("OCR Preview: " + preview.replaceAll("\\n", "\\\\n"));
                    }

                    // Process this page for terms
                    processPageForTerms(pageText, page + 1);

                } catch (Exception e) {
                    String errorMsg = "OCR error on page " + (page + 1) + ": " + e.getMessage();
                    debug("OCR Error: " + errorMsg);
                    extractedText.add("[ERROR: " + errorMsg + "]");
                    if (callback != null) {
                        callback.onError(errorMsg);
                    }
                }
            }

            if (callback != null) {
                callback.onComplete("OCR processing complete");
            }

        } catch (IOException e) {
            String errorMsg = "PDF processing error: " + e.getMessage();
            debug("PDF Error: " + errorMsg);
            if (callback != null) {
                callback.onError(errorMsg);
            }
        }

        return extractedText;
    }

    private boolean hasValidGlossaryStructure(String text) {
        if (text == null || text.length() < 1000) {
            debug("Text too short for glossary: " + (text != null ? text.length() : "null"));
            return false;
        }

        boolean hasEnglish = text.matches(".*[a-zA-Z]{3,}.*");
        boolean hasTibetan = text.matches(".*[\\u0F00-\\u0FFF].*");
        boolean hasStructure = text.split("\\n").length > 10;

        debug("Glossary structure check - English: " + hasEnglish + ", Tibetan: " + hasTibetan + ", Structure: " + hasStructure);

        return hasEnglish && hasTibetan && hasStructure;
    }

    private void processExtractedText(String fullText) {
        debug("Processing direct text extraction...");
        String[] pages = splitIntoPages(fullText);
        debug("Split into " + pages.length + " pages");

        for (int i = 0; i < pages.length; i++) {
            processPageText(pages[i], i + 1);
        }
    }

    private String[] splitIntoPages(String fullText) {
        List<String> pages = new ArrayList<>();
        String[] lines = fullText.split("\\n");
        StringBuilder currentPage = new StringBuilder();

        for (String line : lines) {
            if (isPageBreak(line)) {
                if (currentPage.length() > 0) {
                    pages.add(currentPage.toString());
                    currentPage = new StringBuilder();
                }
            } else {
                currentPage.append(line).append("\\n");
            }
        }

        if (currentPage.length() > 0) {
            pages.add(currentPage.toString());
        }

        return pages.toArray(new String[0]);
    }

    private boolean isPageBreak(String line) {
        return line.matches("^\\s*\\d+\\s*$") ||
               line.matches(".*absolute pressure.*\\d+") ||
               (line.length() < 5 && line.matches("\\s*"));
    }

    private void processPageText(String pageText, int pageNumber) {
        debug("Processing page " + pageNumber + " text (direct extraction)");
        List<TermPair> terms = extractTermsFromText(pageText);
        debug("Extracted " + terms.size() + " terms from page " + pageNumber);

        for (TermPair term : terms) {
            saveTermPair(term, pageNumber, "Text extraction");
        }
    }

    private void processPageForTerms(String pageText, int pageNumber) {
        debug("=== Processing page " + pageNumber + " for terms ===");

        if (pageText == null || pageText.trim().isEmpty()) {
            debug("Page text is empty");
            return;
        }

        List<TermPair> terms = extractTermsFromPage(pageText);
        debug("Found " + terms.size() + " potential terms on page " + pageNumber);

        for (TermPair term : terms) {
            debug("Term: '" + term.englishTerm + "' -> '" + term.tibetanTerm + "' [" + term.subject + "]");
            saveTermPair(term, pageNumber, "OCR extracted");
        }
    }

    private List<TermPair> extractTermsFromPage(String pageText) {
        List<TermPair> terms = new ArrayList<>();
        String cleanedText = cleanOCRText(pageText);

        debug("Cleaned text preview: " + cleanedText.substring(0, Math.min(300, cleanedText.length())));

        // Use line-by-line column parser for proper 4-column separation
        List<LineByLineColumnParser.TermBuilder> parsedTerms = LineByLineColumnParser.parseColumns(cleanedText);
        for (LineByLineColumnParser.TermBuilder builder : parsedTerms) {
            if (builder.isComplete() && isValidTerm(builder.englishTerm, builder.tibetanTerm)) {
                TermPair termPair = new TermPair(builder.englishTerm, builder.tibetanTerm, builder.subject, builder.tibetanDefinition);
                terms.add(termPair);
                debug("✓ Line-parsed: " + builder.englishTerm + " [" + builder.subject + "] -> " + builder.tibetanTerm);
                if (!builder.tibetanDefinition.trim().isEmpty()) {
                    debug("    Definition: " + builder.tibetanDefinition.substring(0, Math.min(50, builder.tibetanDefinition.length())) + "...");
                }
            }
        }

        // Pattern 1: Main glossary format - English + Subject + Tibetan (per Feedback.txt)
        Pattern mainGlossaryPattern = Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s-']+?)\\s+" +           // English term (e.g., "abiotic factor")
            "([a-z]{2,6})\\s+" +                         // Subject code (e.g., "bot", "psycho")
            "([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།]*)", // Tibetan text
            Pattern.MULTILINE
        );
        
        // Pattern 2: Abbreviations format - abbrev. FullTerm Tibetan (pages 18-22)
        Pattern abbreviationsPattern = Pattern.compile(
            "^([a-zA-Z]+\\.)\\s+" +                      // Abbreviation (e.g., "acc.", "chem.")
            "([A-Za-z][A-Za-z\\s-']+?)\\s+" +           // Full English term (e.g., "Accountancy")
            "([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།]*)", // Tibetan text
            Pattern.MULTILINE
        );

        // Try main glossary pattern first
        Matcher matcher = mainGlossaryPattern.matcher(cleanedText);
        while (matcher.find()) {
            String englishTerm = cleanEnglishTerm(matcher.group(1));
            String subject = matcher.group(2);
            String tibetanTerm = cleanTibetanTerm(matcher.group(3));

            if (isValidTerm(englishTerm, tibetanTerm)) {
                terms.add(new TermPair(englishTerm, tibetanTerm, subject));
                debug("Main glossary match: " + englishTerm + " [" + subject + "] -> " + tibetanTerm);
            }
        }
        
        // Try abbreviations pattern
        matcher = abbreviationsPattern.matcher(cleanedText);
        while (matcher.find()) {
            String abbreviation = matcher.group(1);
            String fullTerm = cleanEnglishTerm(matcher.group(2));
            String tibetanTerm = cleanTibetanTerm(matcher.group(3));

            if (isValidTerm(fullTerm, tibetanTerm)) {
                terms.add(new TermPair(fullTerm, tibetanTerm, "abbrev"));
                debug("Abbreviation match: " + abbreviation + " = " + fullTerm + " -> " + tibetanTerm);
            }
        }

        // Pattern 2: Handle OCR failures - English + Subject + Garbled text
        // This is what was working before for cases like "table"
        Pattern ocrFailurePattern = Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s-']+?)\\s+" +           // English term
            "([a-z]+(?:,\\s*[a-z]+)*)\\s+" +             // Subject codes
            "([A-Za-z0-9\\s]{5,})",                       // Garbled text (OCR failed Tibetan)
            Pattern.MULTILINE
        );

        matcher = ocrFailurePattern.matcher(cleanedText);
        while (matcher.find()) {
            String englishTerm = cleanEnglishTerm(matcher.group(1));
            String subjects = matcher.group(2);
            String garbledText = matcher.group(3).trim();

            // Create placeholder for failed Tibetan OCR
            String tibetanPlaceholder = "[Tibetan OCR failed: " +
                garbledText.substring(0, Math.min(20, garbledText.length())) + "...]";

            if (isValidEnglishTerm(englishTerm)) {
                terms.add(new TermPair(englishTerm, tibetanPlaceholder, subjects));
                debug("OCR failure match: " + englishTerm + " [" + subjects + "] -> " + tibetanPlaceholder);
            }
        }

        // Pattern 3: Multi-line terms
        Pattern multilinePattern = Pattern.compile(
            "([a-zA-Z][a-zA-Z\\s-']+?)\\s*\\n" +         // First line English
            "([a-zA-Z][a-zA-Z\\s-']*?)\\s+" +            // Second line English
            "([a-z]+(?:,\\s*[a-z]+)*)\\s+" +             // Subject codes
            "([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།]*)", // Tibetan text
            Pattern.MULTILINE
        );

        matcher = multilinePattern.matcher(cleanedText);
        while (matcher.find()) {
            String englishPart1 = cleanEnglishTerm(matcher.group(1));
            String englishPart2 = cleanEnglishTerm(matcher.group(2));
            String subjects = matcher.group(3);
            String tibetanTerm = cleanTibetanTerm(matcher.group(4));

            String fullEnglishTerm = (englishPart1 + " " + englishPart2).trim();

            if (isValidTerm(fullEnglishTerm, tibetanTerm)) {
                terms.add(new TermPair(fullEnglishTerm, tibetanTerm, subjects));
                debug("Multi-line match: " + fullEnglishTerm + " [" + subjects + "] -> " + tibetanTerm);
            }
        }

        return terms;
    }

    private List<TermPair> extractTermsFromText(String text) {
        List<TermPair> terms = new ArrayList<>();

        Pattern termPattern = Pattern.compile(
            "([a-zA-Z][a-zA-Z\\s-']+?)\\s+" +              // English term
            "([a-z]+(?:[.,]\\s*[a-z]+)*)\\s+" +            // Subject codes
            "([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།]*)", // Tibetan text
            Pattern.MULTILINE
        );

        Matcher matcher = termPattern.matcher(text);
        while (matcher.find()) {
            String englishTerm = cleanEnglishTerm(matcher.group(1));
            String subjects = matcher.group(2);
            String tibetanTerm = cleanTibetanTerm(matcher.group(3));

            if (isValidTerm(englishTerm, tibetanTerm)) {
                terms.add(new TermPair(englishTerm, tibetanTerm, subjects));
            }
        }

        return terms;
    }

    private String cleanOCRText(String text) {
        return text
                .replaceAll("^\\d+\\s*$", "")
                .replaceAll("[─━═-]{3,}", "")
                .replaceAll("\\s{3,}", " ")
                .replaceAll("\\n{3,}", "\\n\\n")
                .trim();
    }

    private String cleanEnglishTerm(String term) {
        return term.trim()
                .replaceAll("\\s+", " ")
                .replaceAll("^[^a-zA-Z]+|[^a-zA-Z\\s'-]+$", "")
                .toLowerCase();
    }

    private String cleanTibetanTerm(String term) {
        return term.trim()
                .replaceAll("\\s+", " ")
                .replaceAll("[།]{2,}$", "།")
                .trim();
    }

    private boolean isValidTerm(String englishTerm, String tibetanTerm) {
        if (englishTerm == null || tibetanTerm == null) return false;
        if (englishTerm.length() < 2 || tibetanTerm.length() < 2) return false;
        if (englishTerm.length() > 100) return false;

        // Allow placeholders for failed OCR
        if (!tibetanTerm.matches(".*[\\u0F00-\\u0FFF].*") &&
            !tibetanTerm.startsWith("[Tibetan OCR failed")) return false;

        if (!englishTerm.matches("[a-zA-Z][a-zA-Z\\s'-]*")) return false;

        return true;
    }

    private boolean isValidEnglishTerm(String englishTerm) {
        if (englishTerm == null || englishTerm.trim().isEmpty()) return false;

        englishTerm = englishTerm.trim().toLowerCase();

        if (englishTerm.length() < 2 || englishTerm.length() > 50) return false;
        if (!englishTerm.matches("[a-zA-Z][a-zA-Z\\s'-]*")) return false;

        // Skip obvious document words
        String[] skipWords = {"abbreviations", "glossary", "terms", "page", "tibetan"};
        for (String skip : skipWords) {
            if (englishTerm.equals(skip)) return false;
        }

        return true;
    }

    private void saveTermPair(TermPair term, int pageNumber, String method) {
        try {
            String context = "Page " + pageNumber + " (" + method + ")";
            if (term.subject != null && !term.subject.isEmpty()) {
                context += " | Subject: " + term.subject;
            }

            // Include Tibetan definition in notes field
            String notes = "Requires manual review";
            if (term.tibetanDefinition != null && !term.tibetanDefinition.trim().isEmpty()) {
                notes = "Definition: " + term.tibetanDefinition.trim();
                if (notes.length() > 200) {
                    notes = notes.substring(0, 200) + "...";
                }
            }

            boolean success = dbManager.addTerm(
                term.englishTerm,
                "English",
                term.tibetanTerm,
                "Tibetan",
                context,
                "OCR-Bot",
                notes
            );

            if (success) {
                debug("✓ Saved: " + term.englishTerm + " -> " + term.tibetanTerm);
                if (!term.tibetanDefinition.isEmpty()) {
                    debug("    Definition: " + term.tibetanDefinition.substring(0, Math.min(50, term.tibetanDefinition.length())) + "...");
                }
            } else {
                debug("✗ Failed to save: " + term.englishTerm);
            }
        } catch (Exception e) {
            debug("✗ Error saving term: " + e.getMessage());
        }
    }

    public String processSinglePage(String pdfPath, int pageNumber) {
        try (PDDocument document = Loader.loadPDF(new File(pdfPath))) {
            PDFRenderer pdfRenderer = new PDFRenderer(document);

            if (pageNumber >= document.getNumberOfPages()) {
                return "Page " + pageNumber + " does not exist.";
            }

            BufferedImage bufferedImage = pdfRenderer.renderImageWithDPI(pageNumber, 400, ImageType.RGB);
            return tesseract.doOCR(bufferedImage);

        } catch (Exception e) {
            return "Error processing page " + pageNumber + ": " + e.getMessage();
        }
    }

    public void setTesseractDataPath(String dataPath) {
        tesseract.setDatapath(dataPath);
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
        String tibetanDefinition = ""; // 4th column - Tibetan definition

        TermPair(String englishTerm, String tibetanTerm, String subject) {
            this.englishTerm = englishTerm;
            this.tibetanTerm = tibetanTerm;
            this.subject = subject;
        }
        
        TermPair(String englishTerm, String tibetanTerm, String subject, String tibetanDefinition) {
            this.englishTerm = englishTerm;
            this.tibetanTerm = tibetanTerm;
            this.subject = subject;
            this.tibetanDefinition = tibetanDefinition != null ? tibetanDefinition : "";
        }
    }
}