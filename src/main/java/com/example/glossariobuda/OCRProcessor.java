package com.example.glossariobuda;

import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;

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
    
    public OCRProcessor(DatabaseManager dbManager) {
        this.dbManager = dbManager;
        this.tesseract = new Tesseract();
        setupTesseract();
    }
    
    private void setupTesseract() {
        // Configure Tesseract for Tibetan text recognition
        // Try bundled tessdata first, then fallback to system installation
        String bundledPath = System.getProperty("user.dir") + "/tessdata";
        String systemPath = "C:\\Program Files\\Tesseract-OCR\\tessdata";
        
        if (new File(bundledPath).exists()) {
            tesseract.setDatapath(bundledPath);
        } else if (new File(systemPath).exists()) {
            tesseract.setDatapath(systemPath);
        } else {
            // Let Tesseract use default path
            System.out.println("Warning: Using default Tesseract path. OCR may not work properly.");
        }
        
        // Check if language files exist and configure accordingly
        String dataPath = systemPath; // Use system path for checking files
        
        boolean hasEnglish = new File(dataPath, "eng.traineddata").exists();
        boolean hasTibetan = new File(dataPath, "bod.traineddata").exists();
        
        if (hasEnglish && hasTibetan) {
            tesseract.setLanguage("eng+bod"); // English + Tibetan
            System.out.println("OCR configured for English + Tibetan");
        } else if (hasEnglish) {
            tesseract.setLanguage("eng"); // English only
            System.out.println("OCR configured for English only (Tibetan language pack not found)");
        } else if (hasTibetan) {
            tesseract.setLanguage("bod"); // Tibetan only
            System.out.println("OCR configured for Tibetan only (English language pack not found)");
        } else {
            // Fallback to default language
            System.out.println("Warning: No language packs found. Using default OCR settings.");
            System.out.println("Please install Tesseract language packs:");
            System.out.println("- eng.traineddata (English)");
            System.out.println("- bod.traineddata (Tibetan)");
            System.out.println("Download from: https://github.com/tesseract-ocr/tessdata");
        }
        
        // Use LSTM engine only (more compatible)
        tesseract.setOcrEngineMode(1); // LSTM engine only (avoid legacy engine issues)
        tesseract.setPageSegMode(3); // Fully automatic page segmentation
        tesseract.setVariable("user_defined_dpi", "300");
        tesseract.setVariable("preserve_interword_spaces", "1");
        
        // Additional error handling variables
        tesseract.setVariable("debug_file", "NUL"); // Suppress debug output on Windows
    }
    
    public List<String> processPDF(String pdfPath, ProgressCallback callback) {
        List<String> extractedText = new ArrayList<>();
        String previousPageEndText = ""; // Store incomplete text from previous page
        
        try (PDDocument document = Loader.loadPDF(new File(pdfPath))) {
            PDFRenderer pdfRenderer = new PDFRenderer(document);
            int totalPages = document.getNumberOfPages();
            
            if (callback != null) {
                callback.onProgress(0, totalPages, "Iniciando processamento OCR...");
            }
            
            for (int page = 0; page < totalPages; page++) {
                try {
                    BufferedImage bufferedImage = pdfRenderer.renderImageWithDPI(page, 300, ImageType.RGB);
                    
                    if (callback != null) {
                        callback.onProgress(page + 1, totalPages, 
                            "Processando página " + (page + 1) + " de " + totalPages);
                    }
                    
                    String pageText;
                    try {
                        pageText = tesseract.doOCR(bufferedImage);
                        if (pageText == null || pageText.trim().isEmpty()) {
                            pageText = "[Empty page or OCR failed]";
                        }
                    } catch (Exception ocrError) {
                        pageText = "[OCR Error: " + ocrError.getMessage() + "]";
                        if (callback != null) {
                            callback.onError("OCR error on page " + (page + 1) + ": " + ocrError.getMessage());
                        }
                    }
                    extractedText.add(pageText);
                    
                    // Handle cross-page text reconstruction
                    String processedText = handleCrossPageText(previousPageEndText, pageText, page + 1);
                    
                    // Extract incomplete text at end of this page for next iteration
                    previousPageEndText = extractIncompleteTextAtPageEnd(pageText);
                    
                    // Process the reconstructed text for terms
                    processPageText(processedText, page + 1);
                    
                } catch (Exception e) {
                    String errorMsg = "Erro OCR na página " + (page + 1) + ": " + e.getMessage();
                    extractedText.add("[ERRO: " + errorMsg + "]");
                    if (callback != null) {
                        callback.onError(errorMsg);
                    }
                    previousPageEndText = ""; // Reset on error
                }
            }
            
            // Process any remaining incomplete text from the last page
            if (!previousPageEndText.trim().isEmpty()) {
                processPageText(previousPageEndText, totalPages);
            }
            
            if (callback != null) {
                callback.onComplete("OCR completo! Processadas " + totalPages + " páginas.");
            }
            
        } catch (IOException e) {
            String errorMsg = "Erro ao processar PDF: " + e.getMessage();
            if (callback != null) {
                callback.onError(errorMsg);
            }
        }
        
        return extractedText;
    }
    
    private void processPageText(String pageText, int pageNumber) {
        // Extract page header information first
        String pageHeader = extractPageHeader(pageText);
        
        // Extract terms with improved pattern matching
        List<TermPair> terms = extractTermsFromText(pageText);
        
        for (TermPair term : terms) {
            // Enhanced context with page header and subject information
            String context = "Página " + pageNumber + " via OCR";
            if (pageHeader != null && !pageHeader.isEmpty()) {
                context += " | " + pageHeader;
            }
            if (term.context != null && !term.context.isEmpty()) {
                context += " | " + term.context;
            }
            
            dbManager.addTerm(
                term.sourceTerm,
                term.sourceLanguage,
                term.targetTerm,
                term.targetLanguage,
                context,
                "OCR-Bot",
                "Necessita revisão manual"
            );
        }
    }
    
    private List<TermPair> extractTermsFromText(String text) {
        List<TermPair> terms = new ArrayList<>();
        
        // Pattern 1: Multi-line English terms with Tibetan
        // Example: "clean bill of    𝑥𝑥     གསོ་བའི་དོན། གཞན་གྱི་སྐུ་ཚབ་དང་བཅས་པའི།"
        //          "health                  ཚོད་འཛིན། མི་འཁྲུག་གི་གསོ་བའི་དོན།"
        Pattern multilinePattern = Pattern.compile(
            "([a-zA-Z][a-zA-Z\\s]+?)\\s+([𝑥𝑥a-zA-Z*]+)\\s+([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།\\r\\n]+?)(?=\\n[a-zA-Z]|\\n\\s*\\n|$)",
            Pattern.MULTILINE | Pattern.DOTALL
        );
        
        Matcher matcher = multilinePattern.matcher(text);
        while (matcher.find()) {
            String englishPart1 = matcher.group(1).trim();
            String marker = matcher.group(2).trim();
            String tibetanBlock = matcher.group(3).trim();
            
            // Look for additional English words on the next line
            String[] lines = tibetanBlock.split("\\r?\\n");
            String englishTerm = englishPart1;
            StringBuilder tibetanText = new StringBuilder();
            
            for (String line : lines) {
                line = line.trim();
                if (line.matches("^[a-zA-Z][a-zA-Z\\s]*$")) {
                    // This line contains English text - add to term
                    englishTerm += " " + line;
                } else if (line.matches(".*[\\u0F00-\\u0FFF].*")) {
                    // This line contains Tibetan text
                    if (tibetanText.length() > 0) tibetanText.append(" ");
                    tibetanText.append(line);
                }
            }
            
            String finalTibetan = tibetanText.toString().replaceAll("[།\\s]+$", "").trim();
            String finalEnglish = englishTerm.trim();
            String context = "Marker: " + marker;
            
            if (!finalEnglish.isEmpty() && !finalTibetan.isEmpty()) {
                terms.add(new TermPair(
                    finalEnglish, "English",
                    finalTibetan, "Tibetan",
                    context
                ));
            }
        }
        
        // Pattern 2: Single-line format with *subject*
        // Example: "absent *psycho* མི་མཚམས་པའི། དབུ་མའི་རྣམ་པ་བཞི་དང་དབྱེ་བ།"
        Pattern singlelinePattern = Pattern.compile(
            "([a-zA-Z][a-zA-Z\\s]+?)\\s+\\*([a-zA-Z]+)\\*\\s+([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།]+)",
            Pattern.MULTILINE | Pattern.DOTALL
        );
        
        matcher = singlelinePattern.matcher(text);
        while (matcher.find()) {
            String englishTerm = matcher.group(1).trim();
            String subject = matcher.group(2).trim();
            String tibetanText = matcher.group(3).trim();
            
            tibetanText = tibetanText.replaceAll("[།\\s]+$", "").trim();
            String context = "Subject: " + subject;
            
            // Avoid duplicates from multiline pattern
            final String finalEnglishTerm = englishTerm;
            final String finalTibetanText = tibetanText;
            boolean isDuplicate = terms.stream().anyMatch(term -> 
                term.sourceTerm.equals(finalEnglishTerm) && term.targetTerm.equals(finalTibetanText)
            );
            
            if (!englishTerm.isEmpty() && !tibetanText.isEmpty() && !isDuplicate) {
                terms.add(new TermPair(
                    englishTerm, "English",
                    tibetanText, "Tibetan",
                    context
                ));
            }
        }
        
        return terms;
    }
    
    private String extractPageHeader(String pageText) {
        // Pattern to extract page header: "dictionary term + spaces + page number"
        // Example: "absolute pressure                                   4"
        Pattern headerPattern = Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s]+?)\\s+([\\d]+)\\s*$",
            Pattern.MULTILINE
        );
        
        // Also look for headers with separator lines
        Pattern headerWithSeparatorPattern = Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s]+?)\\s+([\\d]+)\\s*\\n[─\\-_═]{3,}",
            Pattern.MULTILINE
        );
        
        Matcher matcher = headerWithSeparatorPattern.matcher(pageText);
        if (matcher.find()) {
            String term = matcher.group(1).trim();
            String pageNum = matcher.group(2).trim();
            return "Header: " + term + " (p." + pageNum + ")";
        }
        
        // Fallback to simple header pattern
        matcher = headerPattern.matcher(pageText);
        if (matcher.find()) {
            String term = matcher.group(1).trim();
            String pageNum = matcher.group(2).trim();
            
            // Validate this looks like a header (reasonable term length)
            if (term.length() > 3 && term.length() < 50) {
                return "Header: " + term + " (p." + pageNum + ")";
            }
        }
        
        return null;
    }
    
    private String handleCrossPageText(String previousPageEnd, String currentPageText, int pageNumber) {
        if (previousPageEnd.trim().isEmpty()) {
            return currentPageText;
        }
        
        // Look for continuation at the beginning of current page (after header)
        String currentPageStart = extractTextAfterHeader(currentPageText);
        
        // Try to reconstruct broken terms
        String reconstructed = reconstructBrokenTerms(previousPageEnd, currentPageStart);
        
        // If reconstruction found a complete term, remove the broken parts and add the complete term
        if (reconstructed != null && !reconstructed.isEmpty()) {
            // Remove the incomplete start from current page and prepend the reconstructed term
            String cleanedCurrentPage = removeIncompleteStart(currentPageText);
            return reconstructed + "\n" + cleanedCurrentPage;
        }
        
        // No reconstruction possible, just return current page
        return currentPageText;
    }
    
    private String extractIncompleteTextAtPageEnd(String pageText) {
        String[] lines = pageText.split("\\r?\\n");
        StringBuilder incompleteText = new StringBuilder();
        
        // Look at last few lines to see if they contain incomplete terms
        for (int i = Math.max(0, lines.length - 3); i < lines.length; i++) {
            String line = lines[i].trim();
            
            // Check if line looks incomplete (English without Tibetan, or Tibetan without context)
            if (isIncompleteEnglishTerm(line) || isIncompleteTibetanText(line)) {
                if (incompleteText.length() > 0) incompleteText.append("\n");
                incompleteText.append(line);
            }
        }
        
        return incompleteText.toString();
    }
    
    private String extractTextAfterHeader(String pageText) {
        // Remove header and separator lines to get main content
        String[] lines = pageText.split("\\r?\\n");
        StringBuilder content = new StringBuilder();
        boolean headerPassed = false;
        
        for (String line : lines) {
            if (!headerPassed) {
                // Skip header lines and separator lines
                if (line.matches("^[a-zA-Z][a-zA-Z\\s]+\\s+\\d+\\s*$") || 
                    line.matches("^[─\\-_═]{3,}.*")) {
                    headerPassed = true;
                    continue;
                }
            } else {
                // Take first few lines after header
                if (content.length() > 200) break; // Limit to avoid too much text
                if (content.length() > 0) content.append("\n");
                content.append(line);
            }
        }
        
        return content.toString();
    }
    
    private String reconstructBrokenTerms(String pageEnd, String pageStart) {
        String[] endLines = pageEnd.split("\\r?\\n");
        String[] startLines = pageStart.split("\\r?\\n");
        
        if (endLines.length == 0 || startLines.length == 0) return null;
        
        // Try to combine incomplete English terms
        String lastEndLine = endLines[endLines.length - 1].trim();
        String firstStartLine = startLines[0].trim();
        
        // Case 1: English term split across pages
        if (isIncompleteEnglishTerm(lastEndLine) && isIncompleteEnglishTerm(firstStartLine)) {
            // Look for Tibetan text in following lines
            StringBuilder tibetanText = new StringBuilder();
            for (int i = 1; i < Math.min(startLines.length, 5); i++) {
                if (startLines[i].matches(".*[\\u0F00-\\u0FFF].*")) {
                    if (tibetanText.length() > 0) tibetanText.append(" ");
                    tibetanText.append(startLines[i].trim());
                }
            }
            
            if (tibetanText.length() > 0) {
                String completeEnglish = lastEndLine + " " + firstStartLine;
                return completeEnglish + "   " + tibetanText.toString();
            }
        }
        
        // Case 2: Tibetan text split across pages
        if (isIncompleteTibetanText(lastEndLine) && isIncompleteTibetanText(firstStartLine)) {
            // Look back for English term
            String englishTerm = "";
            for (int i = endLines.length - 2; i >= Math.max(0, endLines.length - 5); i--) {
                if (endLines[i].matches("^[a-zA-Z][a-zA-Z\\s]*$")) {
                    englishTerm = endLines[i].trim();
                    break;
                }
            }
            
            if (!englishTerm.isEmpty()) {
                String completeTibetan = lastEndLine + " " + firstStartLine;
                return englishTerm + "   " + completeTibetan;
            }
        }
        
        return null;
    }
    
    private String removeIncompleteStart(String pageText) {
        String[] lines = pageText.split("\\r?\\n");
        StringBuilder cleaned = new StringBuilder();
        boolean foundComplete = false;
        
        for (String line : lines) {
            if (!foundComplete && (isIncompleteEnglishTerm(line) || isIncompleteTibetanText(line))) {
                continue; // Skip incomplete start lines
            }
            foundComplete = true;
            if (cleaned.length() > 0) cleaned.append("\n");
            cleaned.append(line);
        }
        
        return cleaned.toString();
    }
    
    private boolean isIncompleteEnglishTerm(String line) {
        // English line without Tibetan text and looks like it could continue
        return line.matches("^[a-zA-Z][a-zA-Z\\s]*$") && 
               !line.matches(".*[\\u0F00-\\u0FFF].*") &&
               line.length() > 2 && line.length() < 30;
    }
    
    private boolean isIncompleteTibetanText(String line) {
        // Tibetan text that seems incomplete (no proper ending punctuation)
        return line.matches(".*[\\u0F00-\\u0FFF].*") && 
               !line.matches(".*[།]\\s*$") &&
               line.length() > 3;
    }
    
    private void processTermWithContext(String englishTerm, String tibetanTerm, String context) {
        // This method can be used to add terms with richer context information
        // Currently just for internal processing, actual DB insertion happens in processPageText
    }
    
    private String detectLanguage(String text) {
        // Simple language detection based on character patterns
        if (text.matches(".*[àáâãäèéêëìíîïòóôõöùúûüç].*")) {
            return "Portuguese";
        } else if (text.matches(".*[a-zA-Z].*")) {
            return "English";
        }
        return "Unknown";
    }
    
    private boolean containsTibetan(String text) {
        return text.matches(".*[\\u0F00-\\u0FFF].*");
    }
    
    public String processSinglePage(String pdfPath, int pageNumber) {
        try (PDDocument document = Loader.loadPDF(new File(pdfPath))) {
            PDFRenderer pdfRenderer = new PDFRenderer(document);
            
            if (pageNumber >= document.getNumberOfPages()) {
                return "Página " + pageNumber + " não existe no documento.";
            }
            
            BufferedImage bufferedImage = pdfRenderer.renderImageWithDPI(pageNumber, 300, ImageType.RGB);
            return tesseract.doOCR(bufferedImage);
            
        } catch (Exception e) {
            return "Erro ao processar página " + pageNumber + ": " + e.getMessage();
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
        final String sourceTerm;
        final String sourceLanguage;
        final String targetTerm;
        final String targetLanguage;
        final String context;
        
        TermPair(String sourceTerm, String sourceLanguage, String targetTerm, String targetLanguage) {
            this(sourceTerm, sourceLanguage, targetTerm, targetLanguage, null);
        }
        
        TermPair(String sourceTerm, String sourceLanguage, String targetTerm, String targetLanguage, String context) {
            this.sourceTerm = sourceTerm;
            this.sourceLanguage = sourceLanguage;
            this.targetTerm = targetTerm;
            this.targetLanguage = targetLanguage;
            this.context = context;
        }
    }
}