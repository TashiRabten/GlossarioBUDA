package com.example.glossariobuda;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Coordinate-Based Column Detector for Buddhist Glossary
 * 
 * Uses industry-standard approach of clustering text by x-coordinates
 * to identify column boundaries, then handles multi-line content properly.
 * 
 * This approach is more reliable than fixed character positions because
 * it works with actual PDF text positioning data.
 */
public class CoordinateColumnDetector {
    
    public static class TextBlock {
        public final String text;
        public final float x;
        public final float y;
        public final float width;
        public final float height;
        public final int pageNumber;
        
        public TextBlock(String text, float x, float y, float width, float height, int pageNumber) {
            this.text = text.trim();
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.pageNumber = pageNumber;
        }
        
        public float getEndX() {
            return x + width;
        }
        
        @Override
        public String toString() {
            return String.format("'%s' @(%.1f,%.1f) %dx%.1f", 
                text.length() > 20 ? text.substring(0, 20) + "..." : text, 
                x, y, (int)width, height);
        }
    }
    
    public static class ColumnizedEntry {
        public String englishTerm = "";
        public String subject = "";
        public String tibetanTerm = "";
        public String tibetanDefinition = "";
        public int pageNumber;
        public List<TextBlock> sourceBlocks = new ArrayList<>();
        
        public ColumnizedEntry(int pageNumber) {
            this.pageNumber = pageNumber;
        }
        
        @Override
        public String toString() {
            return String.format("'%s' [%s] -> '%s' | Def: '%s' (Page %d)", 
                englishTerm, subject, tibetanTerm, 
                tibetanDefinition.length() > 50 ? tibetanDefinition.substring(0, 50) + "..." : tibetanDefinition,
                pageNumber);
        }
    }
    
    /**
     * Extract text with coordinates from PDF and detect columns
     */
    public static List<ColumnizedEntry> detectColumnsFromPDF(String pdfPath) throws IOException {
        List<ColumnizedEntry> allEntries = new ArrayList<>();
        
        try (PDDocument document = Loader.loadPDF(new File(pdfPath))) {
            for (int pageNum = 1; pageNum <= document.getNumberOfPages(); pageNum++) {
                System.out.println("[COORD] Processing page " + pageNum);
                
                // Extract text blocks with coordinates
                List<TextBlock> textBlocks = extractTextBlocks(document, pageNum);
                
                // Detect column boundaries for this page
                List<Float> columnBoundaries = detectColumnBoundaries(textBlocks);
                
                // Group text blocks into columns
                List<ColumnizedEntry> pageEntries = groupIntoColumns(textBlocks, columnBoundaries, pageNum);
                
                allEntries.addAll(pageEntries);
                
                System.out.println("[COORD] Page " + pageNum + " found " + pageEntries.size() + " entries");
            }
        }
        
        return allEntries;
    }
    
    /**
     * Extract text blocks with precise coordinates using PDFBox
     */
    private static List<TextBlock> extractTextBlocks(PDDocument document, int pageNum) throws IOException {
        List<TextBlock> textBlocks = new ArrayList<>();
        
        PDFTextStripper stripper = new PDFTextStripper() {
            @Override
            protected void writeString(String string, List<TextPosition> textPositions) throws IOException {
                if (textPositions.isEmpty() || string.trim().isEmpty()) return;
                
                // Get bounds of this text string
                TextPosition first = textPositions.get(0);
                TextPosition last = textPositions.get(textPositions.size() - 1);
                
                float x = first.getXDirAdj();
                float y = first.getYDirAdj();
                float width = last.getXDirAdj() + last.getWidthDirAdj() - x;
                float height = first.getHeightDir();
                
                textBlocks.add(new TextBlock(string, x, y, width, height, pageNum));
                System.out.println("[COORD] Block: '" + string + "' at (" + x + "," + y + ")");
            }
        };
        
        stripper.setStartPage(pageNum);
        stripper.setEndPage(pageNum);
        stripper.setSortByPosition(true); // Enable position sorting
        
        try {
            String extractedText = stripper.getText(document); // This triggers the writeString calls
            System.out.println("[COORD] Raw extracted text: " + extractedText.substring(0, Math.min(200, extractedText.length())) + "...");
        } catch (Exception e) {
            System.out.println("[COORD] Text extraction failed: " + e.getMessage());
        }
        
        // Sort by Y position (top to bottom), then X position (left to right)
        textBlocks.sort((a, b) -> {
            int yCompare = Float.compare(a.y, b.y);
            if (Math.abs(a.y - b.y) < 2.0f) { // Same line tolerance
                return Float.compare(a.x, b.x);
            }
            return yCompare;
        });
        
        System.out.println("[COORD] Extracted " + textBlocks.size() + " text blocks from page " + pageNum);
        
        // If no text blocks found, this might be an image-based PDF
        if (textBlocks.isEmpty()) {
            System.out.println("[COORD] WARNING: No text blocks found - PDF likely contains scanned images rather than selectable text");
        }
        
        return textBlocks;
    }
    
    /**
     * Detect column boundaries by clustering x-coordinates
     */
    private static List<Float> detectColumnBoundaries(List<TextBlock> textBlocks) {
        System.out.println("[COORD] Detecting column boundaries...");
        
        // Collect all x-coordinates
        List<Float> xCoordinates = textBlocks.stream()
            .map(block -> block.x)
            .distinct()
            .sorted()
            .collect(Collectors.toList());
        
        System.out.println("[COORD] Found " + xCoordinates.size() + " unique x-coordinates");
        
        // Cluster x-coordinates to find column starts
        List<Float> columnStarts = clusterCoordinates(xCoordinates);
        
        System.out.println("[COORD] Detected " + columnStarts.size() + " column boundaries: " + columnStarts);
        return columnStarts;
    }
    
    /**
     * Cluster coordinates to find natural column boundaries
     */
    private static List<Float> clusterCoordinates(List<Float> coordinates) {
        if (coordinates.isEmpty()) return new ArrayList<>();
        
        List<Float> clusters = new ArrayList<>();
        clusters.add(coordinates.get(0));
        
        final float MIN_COLUMN_GAP = 30.0f; // Minimum gap between columns in PDF points
        
        for (int i = 1; i < coordinates.size(); i++) {
            float current = coordinates.get(i);
            float lastCluster = clusters.get(clusters.size() - 1);
            
            if (current - lastCluster > MIN_COLUMN_GAP) {
                clusters.add(current);
                System.out.println("[CLUSTER] New column at x=" + current + " (gap: " + (current - lastCluster) + ")");
            }
        }
        
        return clusters;
    }
    
    /**
     * Group text blocks into columns and reconstruct entries
     */
    private static List<ColumnizedEntry> groupIntoColumns(List<TextBlock> textBlocks, 
                                                          List<Float> columnBoundaries, 
                                                          int pageNumber) {
        System.out.println("[COORD] Grouping " + textBlocks.size() + " blocks into columns...");
        
        List<ColumnizedEntry> entries = new ArrayList<>();
        
        // Group blocks by rows (similar Y coordinates)
        Map<Integer, List<TextBlock>> rows = groupBlocksByRows(textBlocks);
        
        System.out.println("[COORD] Found " + rows.size() + " rows on page " + pageNumber);
        
        // Process each row to extract column data
        for (Map.Entry<Integer, List<TextBlock>> rowEntry : rows.entrySet()) {
            List<TextBlock> rowBlocks = rowEntry.getValue();
            
            // Sort blocks in this row by x-coordinate
            rowBlocks.sort(Comparator.comparing(block -> block.x));
            
            ColumnizedEntry entry = processRowIntoEntry(rowBlocks, columnBoundaries, pageNumber);
            if (entry != null && isValidEntry(entry)) {
                entries.add(entry);
                System.out.println("[COORD] Created entry: " + entry);
            }
        }
        
        return entries;
    }
    
    /**
     * Group text blocks by rows using Y coordinate clustering
     */
    private static Map<Integer, List<TextBlock>> groupBlocksByRows(List<TextBlock> textBlocks) {
        Map<Integer, List<TextBlock>> rows = new TreeMap<>();
        
        final float ROW_TOLERANCE = 3.0f; // Tolerance for same row in PDF points
        int rowIndex = 0;
        
        for (TextBlock block : textBlocks) {
            boolean addedToRow = false;
            
            // Try to add to existing row
            for (Map.Entry<Integer, List<TextBlock>> entry : rows.entrySet()) {
                List<TextBlock> rowBlocks = entry.getValue();
                if (!rowBlocks.isEmpty()) {
                    float rowY = rowBlocks.get(0).y;
                    if (Math.abs(block.y - rowY) <= ROW_TOLERANCE) {
                        rowBlocks.add(block);
                        addedToRow = true;
                        break;
                    }
                }
            }
            
            // Create new row if not added to existing
            if (!addedToRow) {
                List<TextBlock> newRow = new ArrayList<>();
                newRow.add(block);
                rows.put(rowIndex++, newRow);
            }
        }
        
        return rows;
    }
    
    /**
     * Process a row of text blocks into a column entry
     */
    private static ColumnizedEntry processRowIntoEntry(List<TextBlock> rowBlocks, 
                                                       List<Float> columnBoundaries, 
                                                       int pageNumber) {
        if (rowBlocks.isEmpty()) return null;
        
        ColumnizedEntry entry = new ColumnizedEntry(pageNumber);
        entry.sourceBlocks = new ArrayList<>(rowBlocks);
        
        // Assign blocks to columns based on their x-coordinates
        for (TextBlock block : rowBlocks) {
            int columnIndex = determineColumn(block.x, columnBoundaries);
            String cleanText = cleanBlockText(block.text);
            
            switch (columnIndex) {
                case 0: // English column
                    if (!cleanText.isEmpty() && isEnglishText(cleanText)) {
                        entry.englishTerm += (entry.englishTerm.isEmpty() ? "" : " ") + cleanText;
                    }
                    break;
                case 1: // Subject column
                    if (!cleanText.isEmpty() && isSubjectCode(cleanText)) {
                        entry.subject += (entry.subject.isEmpty() ? "" : " ") + cleanText;
                    }
                    break;
                case 2: // Tibetan Term column
                    if (!cleanText.isEmpty() && containsTibetanChars(cleanText)) {
                        entry.tibetanTerm += (entry.tibetanTerm.isEmpty() ? "" : " ") + cleanText;
                    }
                    break;
                case 3: // Tibetan Definition column
                default:
                    if (!cleanText.isEmpty()) {
                        entry.tibetanDefinition += (entry.tibetanDefinition.isEmpty() ? "" : " ") + cleanText;
                    }
                    break;
            }
        }
        
        // Post-process: split Tibetan content if it ended up in wrong columns
        redistributeTibetanContent(entry);
        
        return entry;
    }
    
    /**
     * Determine which column a text block belongs to
     */
    private static int determineColumn(float x, List<Float> columnBoundaries) {
        for (int i = 0; i < columnBoundaries.size(); i++) {
            if (x < columnBoundaries.get(i) + 20.0f) { // 20pt tolerance
                return i;
            }
        }
        return columnBoundaries.size(); // Last column
    }
    
    /**
     * Redistribute Tibetan content that may have been misclassified
     */
    private static void redistributeTibetanContent(ColumnizedEntry entry) {
        // If Tibetan content ended up in term column, try to split it
        if (!entry.tibetanTerm.isEmpty() && entry.tibetanDefinition.isEmpty()) {
            String[] parts = smartSplitTibetan(entry.tibetanTerm);
            entry.tibetanTerm = parts[0];
            entry.tibetanDefinition = parts[1];
        }
        
        // Clean up any English text that leaked into Tibetan columns
        entry.tibetanTerm = removeEnglishFromTibetan(entry.tibetanTerm);
        entry.tibetanDefinition = removeEnglishFromTibetan(entry.tibetanDefinition);
    }
    
    // Helper methods
    private static String cleanBlockText(String text) {
        return text.replaceAll("\\s+", " ").trim();
    }
    
    private static boolean isEnglishText(String text) {
        return text.matches("^[a-zA-Z][a-zA-Z\\s'-]*$");
    }
    
    private static boolean isSubjectCode(String text) {
        return text.matches("^[a-z]{2,6}$");
    }
    
    private static boolean containsTibetanChars(String text) {
        return text.chars().anyMatch(c -> c >= 0x0F00 && c <= 0x0FFF);
    }
    
    private static String[] smartSplitTibetan(String tibetanText) {
        // Use the same logic from ColumnDetector
        String cleaned = tibetanText.trim();
        
        // Look for རྐྱེན། completion
        int rkyenIndex = cleaned.indexOf("རྐྱེན།");
        if (rkyenIndex >= 0) {
            String term = cleaned.substring(0, rkyenIndex + 5);
            String definition = cleaned.substring(rkyenIndex + 5).trim();
            return new String[]{term, definition};
        }
        
        // Split on first ། punctuation
        int shadIndex = cleaned.indexOf('།');
        if (shadIndex > 0 && shadIndex < cleaned.length() - 3) {
            String term = cleaned.substring(0, shadIndex + 1);
            String definition = cleaned.substring(shadIndex + 1).trim();
            return new String[]{term, definition};
        }
        
        // Fallback: split roughly in middle
        String[] words = cleaned.split("\\s+");
        if (words.length > 4) {
            int splitPoint = Math.min(3, words.length / 3);
            String term = String.join(" ", Arrays.copyOfRange(words, 0, splitPoint));
            String definition = String.join(" ", Arrays.copyOfRange(words, splitPoint, words.length));
            return new String[]{term, definition};
        }
        
        return new String[]{cleaned, ""};
    }
    
    private static String removeEnglishFromTibetan(String text) {
        return text.replaceAll("\\b[a-zA-Z]+\\b", "").replaceAll("\\s+", " ").trim();
    }
    
    private static boolean isValidEntry(ColumnizedEntry entry) {
        return !entry.englishTerm.trim().isEmpty() && 
               !entry.tibetanTerm.trim().isEmpty() &&
               entry.englishTerm.length() >= 3;
    }
}