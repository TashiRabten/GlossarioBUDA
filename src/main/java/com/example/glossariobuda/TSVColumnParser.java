package com.example.glossariobuda;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class TSVColumnParser {
    private static final Set<String> VALID_SUBJECT_CODES = Set.of(
            "acc", "agri", "anth", "arch", "astro", "bio", "bot", "buddh", "chem",
            "econ", "geo", "geog", "geom", "hist", "ling", "lit", "log", "math",
            "med", "mus", "phil", "phys", "psycho", "zool", "poly", "soc", "rel",
            "adm", "pol", "com", "edu", "cine", "hay" // include edge subject variants
    );

    private static final Pattern TIBETAN_PATTERN = Pattern.compile(".*[\u0F00-\u0FFF].*");

    public static void integrateIntoDatabase(File tsvFile, DatabaseManager dbManager, int pageNumber) {
        System.out.println("[TSV DEBUG] Parsing OCR TSV with coordinate clustering for page " + pageNumber);
        List<OCRWord> words = new ArrayList<>();

        // First, extract OCR words with coordinates
        try (BufferedReader reader = new BufferedReader(new FileReader(tsvFile))) {
            String headerLine = reader.readLine(); // Skip header
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\t");
                if (parts.length >= 12 && !parts[11].trim().isEmpty()) {
                    try {
                        OCRWord word = new OCRWord(
                                Integer.parseInt(parts[0]), // level
                                Integer.parseInt(parts[6]), // left (x coordinate)
                                Integer.parseInt(parts[7]), // top (y coordinate)
                                Integer.parseInt(parts[8]), // width
                                Integer.parseInt(parts[9]), // height
                                parts[11].trim() // text
                        );
                        if (word.level == 5) { // Only individual words
                            words.add(word);
                        }
                    } catch (NumberFormatException e) {
                        continue;
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("[TSV ERROR] Read failed: " + e.getMessage());
            return;
        }

        System.out.println("[TSV DEBUG] Found " + words.size() + " OCR words");

        // Group words into lines
        Map<Integer, List<OCRWord>> lineGroups = groupWordsByLines(words);
        System.out.println("[TSV DEBUG] Grouped into " + lineGroups.size() + " lines");

        // Process each line individually first, then merge related entries
        List<FourColumnEntry> entries = processLinesIndividually(lineGroups);
        
        // Save reconstructed entries to database
        for (FourColumnEntry entry : entries) {
            System.out.println("[TSV DEBUG] Extracted 4-column entry: " + entry);
            saveGlossaryEntry(dbManager, entry.english, entry.subject, entry.tibetanTerm, entry.tibetanDefinition);
        }

        System.out.println("[TSV DEBUG] TSV parsing completed for page " + pageNumber);
    }

    /**
     * Groups OCR words into lines based on Y coordinate proximity
     */
    private static Map<Integer, List<OCRWord>> groupWordsByLines(List<OCRWord> words) {
        Map<Integer, List<OCRWord>> lineGroups = new TreeMap<>();

        for (OCRWord word : words) {
            // Group words with similar Y coordinates (within 30 pixels) into same line
            int lineKey = (word.top / 30) * 30;
            lineGroups.computeIfAbsent(lineKey, k -> new ArrayList<>()).add(word);
        }

        // Sort words within each line by X coordinate (left to right)
        for (List<OCRWord> lineWords : lineGroups.values()) {
            lineWords.sort(Comparator.comparingInt(w -> w.left));
        }

        return lineGroups;
    }

    /**
     * Reconstructs a line of text with appropriate whitespace based on word positions
     */
    private static String reconstructLineWithWhitespace(List<OCRWord> lineWords) {
        if (lineWords.isEmpty()) return "";

        StringBuilder line = new StringBuilder();
        OCRWord previousWord = null;

        for (OCRWord word : lineWords) {
            if (previousWord != null) {
                // Calculate gap between words
                int gap = word.left - (previousWord.left + previousWord.width);

                // Add whitespace based on gap size
                if (gap > 100) {
                    // Large gap - likely column separator
                    line.append("    "); // 4 spaces for major column separation
                } else if (gap > 50) {
                    // Medium gap - sub-column or emphasis
                    line.append("  "); // 2 spaces
                } else if (gap > 10) {
                    // Small gap - normal word spacing
                    line.append(" ");
                }
                // Very small gaps (<=10) - probably connected text, no space
            }

            line.append(word.text);
            previousWord = word;
        }

        return line.toString();
    }

    /**
     * Detects column boundaries by analyzing whitespace patterns across multiple lines
     */
    private static List<ColumnBoundary> detectColumnBoundariesFromLines(List<String> lines) {
        if (lines.isEmpty()) return new ArrayList<>();

        int maxLength = lines.stream().mapToInt(String::length).max().orElse(0);
        int[] whitespaceCount = new int[maxLength];

        // Count whitespace occurrences at each position across all lines
        for (String line : lines) {
            char[] chars = line.toCharArray();
            for (int i = 0; i < chars.length && i < maxLength; i++) {
                if (Character.isWhitespace(chars[i])) {
                    whitespaceCount[i]++;
                }
            }
        }

        // Find positions where whitespace occurs consistently (column boundaries)
        List<ColumnBoundary> boundaries = new ArrayList<>();
        double threshold = Math.max(1, lines.size() * 0.4); // 40% of lines must have whitespace
        boolean inBoundary = false;
        int boundaryStart = -1;

        for (int i = 0; i < whitespaceCount.length; i++) {
            if (!inBoundary && whitespaceCount[i] >= threshold) {
                inBoundary = true;
                boundaryStart = i;
            } else if (inBoundary && whitespaceCount[i] < threshold) {
                if (i - boundaryStart >= 2) { // Minimum boundary width of 2 characters
                    boundaries.add(new ColumnBoundary(boundaryStart, i));
                    System.out.println("[TSV DEBUG] Found column boundary: " + boundaryStart + "-" + i);
                }
                inBoundary = false;
            }
        }

        // Handle boundary that extends to end of line
        if (inBoundary && whitespaceCount.length - boundaryStart >= 2) {
            boundaries.add(new ColumnBoundary(boundaryStart, whitespaceCount.length));
        }

        return boundaries;
    }

    /**
     * Parse a line using detected whitespace boundaries
     */
    private static String[] parseLineWithWhitespaceBoundaries(String line, List<ColumnBoundary> boundaries) {
        if (boundaries.isEmpty()) {
            // Fallback: split on multiple whitespace
            return line.split("\\s{2,}");
        }

        List<String> columns = new ArrayList<>();
        int lastEnd = 0;

        for (ColumnBoundary boundary : boundaries) {
            // Extract column content before this boundary
            if (boundary.start > lastEnd) {
                String columnContent = line.substring(lastEnd, Math.min(boundary.start, line.length())).trim();
                if (!columnContent.isEmpty()) {
                    columns.add(columnContent);
                }
            }
            lastEnd = boundary.end;
        }

        // Add final column after last boundary
        if (lastEnd < line.length()) {
            String finalColumn = line.substring(lastEnd).trim();
            if (!finalColumn.isEmpty()) {
                columns.add(finalColumn);
            }
        }

        return columns.toArray(new String[0]);
    }

    /**
     * Process lines individually respecting coordinate boundaries, then merge entries
     */
    private static List<FourColumnEntry> processLinesIndividually(Map<Integer, List<OCRWord>> lineGroups) {
        List<FourColumnEntry> entries = new ArrayList<>();
        List<Integer> sortedLineKeys = new ArrayList<>(lineGroups.keySet());
        Collections.sort(sortedLineKeys);
        
        FourColumnEntry currentEntry = null;
        Integer previousLineKey = null;
        
        for (Integer lineKey : sortedLineKeys) {
            List<OCRWord> lineWords = lineGroups.get(lineKey);
            if (lineWords.isEmpty()) continue;
            
            // Extract columns from this line respecting coordinate boundaries
            FourColumnEntry lineEntry = extract4ColumnsFromCoordinates(lineWords);
            
            // Check if this should start a new entry
            boolean shouldStartNewEntry = false;
            
            if (lineEntry != null && !lineEntry.english.isEmpty() && !lineEntry.subject.isEmpty()) {
                // Line has both English and subject - definitely new entry
                shouldStartNewEntry = true;
            } else if (lineEntry != null && !lineEntry.english.isEmpty() && currentEntry != null && previousLineKey != null) {
                // Line has English but no subject - check Y-gap and context to determine if it's new entry
                int yGap = lineKey - previousLineKey;
                
                // More aggressive entry detection: smaller Y-gap threshold for better separation
                if (yGap > 90) {
                    shouldStartNewEntry = true;
                    System.out.println("[TSV DEBUG] Y-gap detected (" + yGap + "px) - treating as new entry: " + lineEntry.english);
                } else {
                    System.out.println("[TSV DEBUG] Small Y-gap (" + yGap + "px) - treating as continuation: " + lineEntry.english);
                }
            } else if (lineEntry != null && !lineEntry.english.isEmpty() && currentEntry == null) {
                // First line with English text - start new entry even without subject
                shouldStartNewEntry = true;
            }
            
            previousLineKey = lineKey;
            
            if (shouldStartNewEntry) {
                // This line starts a new entry - save previous if exists
                if (currentEntry != null) {
                    entries.add(currentEntry);
                }
                currentEntry = lineEntry;
                System.out.println("[TSV DEBUG] New entry started: " + currentEntry.english + " | " + currentEntry.subject);
            } else if (currentEntry != null) {
                // This line continues current entry - merge by coordinate columns
                String englishContinuation = "";
                String subjectContinuation = "";
                String termContinuation = "";
                String defContinuation = "";
                
                // Process words in original TSV order preserving sequence
                List<OCRWord> tibetanWords = new ArrayList<>();
                
                for (OCRWord word : lineWords) {
                    String cleanWord = cleanOCRNoise(word.text);
                    if (cleanWord.isEmpty()) continue;
                    
                    if (word.left < 850) {
                        // Column 1: English continuation
                        if (englishContinuation.length() > 0) englishContinuation += " ";
                        englishContinuation += cleanWord;
                    } else if (word.left < 1100) {
                        // Column 2: Subject continuation
                        if (subjectContinuation.length() > 0) subjectContinuation += " ";
                        subjectContinuation += cleanWord;
                    } else if (word.left >= 1100) {
                        // Columns 3&4: Collect Tibetan words preserving TSV order
                        tibetanWords.add(word);
                    }
                }
                
                // Process Tibetan words in exact TSV sequence to preserve syllable order
                for (OCRWord tibetanWord : tibetanWords) {
                    String cleanWord = cleanOCRNoise(tibetanWord.text);
                    if (cleanWord.isEmpty()) continue;
                    
                    if (tibetanWord.left < 1460) {
                        // Column 3: Term continuation
                        termContinuation += cleanWord;
                    } else {
                        // Column 4: Definition continuation  
                        defContinuation += cleanWord;
                    }
                }
                
                // Merge continuations for all columns
                String mergedEnglish = currentEntry.english;
                if (!englishContinuation.isEmpty()) {
                    mergedEnglish = mergedEnglish.isEmpty() ? englishContinuation : mergedEnglish + " " + englishContinuation;
                }
                
                String mergedSubject = currentEntry.subject;
                if (!subjectContinuation.isEmpty()) {
                    String normalizedNewSubject = normalizeSubjectCode(subjectContinuation);
                    if (!normalizedNewSubject.isEmpty()) {
                        mergedSubject = normalizedNewSubject; // Use the new valid subject
                    }
                }
                
                currentEntry = new FourColumnEntry(
                    mergedEnglish,
                    mergedSubject,
                    currentEntry.tibetanTerm + termContinuation,
                    currentEntry.tibetanDefinition + defContinuation
                );
                
                System.out.println("[TSV DEBUG] Merged continuation - Eng: +" + englishContinuation + ", Subj: +" + subjectContinuation + ", Term: +" + termContinuation + ", Def: +" + defContinuation);
            }
        }
        
        // Add final entry
        if (currentEntry != null) {
            entries.add(currentEntry);
        }
        
        return entries;
    }


    /**
     * Extract 4 columns from OCR words using x-coordinate clustering
     */
    private static FourColumnEntry extract4ColumnsFromCoordinates(List<OCRWord> lineWords) {
        // Based on actual TSV coordinates analysis:
        // Column 1 (English): x < 850
        // Column 2 (Subject): 850 <= x < 1100  
        // Column 3 (Tibetan Term): 1100 <= x < 1400 (first part of Tibetan content)
        // Column 4 (Tibetan Definition): x >= 1460 (but merge with Column 3 for single-line entries)
        
        StringBuilder english = new StringBuilder();
        StringBuilder subject = new StringBuilder();
        StringBuilder tibetanTerm = new StringBuilder();
        StringBuilder tibetanDefinition = new StringBuilder();
        
        // Separate processing for non-Tibetan and Tibetan columns to preserve order
        List<OCRWord> tibetanWords = new ArrayList<>();
        
        for (OCRWord word : lineWords) {
            // Clean OCR noise
            String cleanWord = cleanOCRNoise(word.text);
            if (cleanWord.isEmpty()) continue;
            
            if (word.left < 850) {
                // Column 1: English
                if (english.length() > 0) english.append(" ");
                english.append(cleanWord);
            } else if (word.left < 1100) {
                // Column 2: Subject code
                if (subject.length() > 0) subject.append(" ");
                subject.append(cleanWord);
            } else if (word.left >= 1100) {
                // Columns 3&4: Collect Tibetan words for ordered processing
                tibetanWords.add(word);
            }
        }
        
        // Process Tibetan words by actual coordinate ranges
        for (OCRWord tibetanWord : tibetanWords) {
            String cleanWord = cleanOCRNoise(tibetanWord.text);
            if (cleanWord.isEmpty()) continue;
            
            if (tibetanWord.left < 1400) {
                // Column 3: Tibetan term (x=1100-1400)
                tibetanTerm.append(cleanWord);
            } else if (tibetanWord.left >= 1460) {
                // Column 4: Tibetan definition (x>=1460)
                tibetanDefinition.append(cleanWord);
            }
            // Skip words in transition zone 1400-1460
        }
        
        // If no term was found but we have definition, don't artificially split
        // Some entries like "abiotic factor" only have definition in the TSV
        
        String englishText = english.toString().trim();
        String subjectText = normalizeSubjectCode(subject.toString().trim());
        
        
        // Clean and truncate Tibetan term at natural boundaries
        String tibetanTermText = cleanAndTruncateTerm(tibetanTerm.toString().trim());
        String tibetanDefText = cleanDefinitionText(tibetanDefinition.toString().trim());
        
        // Validate we have minimum required content (more lenient - allow missing subject initially)
        if (englishText.isEmpty()) {
            System.out.println("[TSV DEBUG] Skipping line - missing English (" + englishText + ")");
            return null;
        }
        
        return new FourColumnEntry(englishText, subjectText, tibetanTermText, tibetanDefText);
    }

    /**
     * Save a glossary entry to the database with 4-column structure
     */
    private static void saveGlossaryEntry(DatabaseManager db, String englishTerm, String subjectCode, String tibetanTerm, String tibetanDefinition) {
        if (subjectCode.isEmpty() || !VALID_SUBJECT_CODES.contains(subjectCode)) {
            System.out.println("[TSV WARNING] Skipped entry with invalid subject: " + englishTerm + " (subject: '" + subjectCode + "')");
            return;
        }

        // Check for duplicates
        List<DatabaseManager.Term> existing = db.searchTerms(englishTerm);
        for (DatabaseManager.Term t : existing) {
            if (t.getTargetTerm().equals(tibetanTerm)) {
                System.out.println("[TSV DEBUG] Duplicate found, skipping: " + englishTerm);
                return;
            }
        }

        System.out.println("[TSV DEBUG] Saving 4-column entry to DB:");
        System.out.println("  English: " + englishTerm);
        System.out.println("  Subject: " + subjectCode);
        System.out.println("  Tibetan Term: " + tibetanTerm);
        System.out.println("  Tibetan Def: " + (tibetanDefinition.length() > 50 ? tibetanDefinition.substring(0, 50) + "..." : tibetanDefinition));
        
        db.addTerm(englishTerm, "en", tibetanTerm, "bo", subjectCode, null, tibetanDefinition);
    }

    /**
     * Clean OCR noise from text including Latin artifacts in Tibetan
     */
    private static String cleanOCRNoise(String text) {
        if (text == null) return "";
        
        // Remove obvious OCR noise including isolated 'x' characters
        String cleaned = text
            .replaceAll("\\s+x\\s*$", "") // Remove trailing 'x'
            .replaceAll("\\s+x\\s+", " ") // Remove isolated 'x' in middle
            .replaceAll("^x\\s+", "") // Remove leading 'x'
            .replaceAll("\\bx\\b", "") // Remove standalone 'x'
            // Remove OCR artifacts from coordinate data
            .replaceAll("\\bBNA\\b", "") // Remove BNA artifact
            .replaceAll("\\bANA\\b", "") // Remove ANA artifact  
            .replaceAll("\\baT\\b", "") // Remove aT artifact
            .replaceAll("\\bAQ\\b", "") // Remove AQ artifact
            .replaceAll("\\bSTARE\\b", "") // Remove STARE artifact
            .replaceAll("\\bATS\\b", "") // Remove ATS artifact
            .replaceAll("\\bAAS\\b", "") // Remove AAS artifact
            .trim();
            
        // For Tibetan text, remove Latin character artifacts
        if (containsTibetanChars(cleaned)) {
            cleaned = cleanLatinArtifactsFromTibetan(cleaned);
        }
            
        return cleaned;
    }
    
    /**
     * Clean and truncate Tibetan term at natural punctuation boundaries
     */
    private static String cleanAndTruncateTerm(String termText) {
        if (termText == null || termText.isEmpty()) return "";
        
        // Remove obvious OCR noise first
        String cleaned = termText
            .replaceAll("\\bx\\b", "")
            .replaceAll("\\s+", "")
            .trim();
        
        // Find the first Tibetan period (།) - this usually marks end of term
        int periodIndex = cleaned.indexOf("།");
        if (periodIndex > 0 && periodIndex < 50) {
            // Use content up to and including the period
            return cleaned.substring(0, periodIndex + 1);
        }
        
        // If no period, look for other natural break points
        // Truncate at reasonable length (around 25-30 characters for terms)
        if (cleaned.length() > 30) {
            // Look for a tsheg (་) around the 20-30 char range
            for (int i = 20; i < Math.min(30, cleaned.length()); i++) {
                if (cleaned.charAt(i) == '་') {
                    return cleaned.substring(0, i + 1);
                }
            }
            // Fallback: hard truncate at 25 chars
            return cleaned.substring(0, 25) + "...";
        }
        
        // Add period if term doesn't end with proper punctuation
        if (!cleaned.isEmpty() && !cleaned.endsWith("།") && !cleaned.endsWith("་")) {
            return cleaned + "།";
        }
        
        return cleaned;
    }

    /**
     * Clean definition text from OCR noise and formatting issues
     */
    private static String cleanDefinitionText(String defText) {
        if (defText == null || defText.isEmpty()) return "";
        
        String cleaned = defText
            // Remove obvious OCR noise
            .replaceAll("\\bx\\b", "")
            .replaceAll("[0-9]+", "") // Remove numbers that are likely OCR artifacts
            .replaceAll("[A-Z]{2,}", "") // Remove sequences like "ATSAAS", "AQSTARE"
            .replaceAll("['']", "") // Remove stray quotes
            // Clean up spacing
            .replaceAll("\\s+", " ")
            .trim();
        
        // Remove content that looks like OCR errors or page artifacts
        cleaned = cleaned.replaceAll("(ས་སུ་|་ཤོས་ཤོག|ATS|AAS|STARE|AQ)", "");
        
        // Clean Latin artifacts from Tibetan definition text
        cleaned = cleanLatinArtifactsFromTibetan(cleaned);
        
        return cleaned.trim();
    }

    /**
     * Normalize subject codes - handle compound codes like "com, econ"
     */
    private static String normalizeSubjectCode(String rawSubject) {
        if (rawSubject == null || rawSubject.isEmpty()) return "";
        
        String cleaned = rawSubject.toLowerCase().replaceAll("[^a-z,\\s]", "").trim();
        
        // Handle compound subject codes - take the last valid one
        if (cleaned.contains(",")) {
            String[] parts = cleaned.split("[,\\s]+");
            for (int i = parts.length - 1; i >= 0; i--) {
                String part = parts[i].trim();
                if (VALID_SUBJECT_CODES.contains(part)) {
                    return part;
                }
            }
        }
        
        // Single subject code
        return VALID_SUBJECT_CODES.contains(cleaned) ? cleaned : "";
    }
    
    private static String appendText(String base, String next) {
        if (base == null || base.trim().isEmpty()) {
            return next;
        }
        return base + " " + next;
    }

    /**
     * Helper class to represent column boundaries
     */
    static class ColumnBoundary {
        final int start;
        final int end;

        ColumnBoundary(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return String.format("ColumnBoundary[%d-%d]", start, end);
        }
    }

    /**
     * Represents an OCR word with its position and text
     */
    static class OCRWord {
        final int level;
        final int left;
        final int top;
        final int width;
        final int height;
        final String text;

        OCRWord(int level, int left, int top, int width, int height, String text) {
            this.level = level;
            this.left = left;
            this.top = top;
            this.width = width;
            this.height = height;
            this.text = text;
        }

        @Override
        public String toString() {
            return String.format("OCRWord[%s at (%d,%d)]", text, left, top);
        }
    }

    /**
     * Clean Latin character artifacts from Tibetan text
     */
    private static String cleanLatinArtifactsFromTibetan(String text) {
        if (text == null || text.isEmpty()) return "";
        
        return text
            // Remove common Latin artifacts found in OCR
            .replaceAll("'a", "")     // Remove 'a artifacts
            .replaceAll("'A", "")     // Remove 'A artifacts  
            .replaceAll("\\ba\\b", "")  // Remove standalone 'a'
            .replaceAll("\\bA\\b", "")  // Remove standalone 'A'
            .replaceAll("[a-zA-Z]\\'", "") // Remove letter+apostrophe combinations
            .replaceAll("\\'[a-zA-Z]", "") // Remove apostrophe+letter combinations
            .replaceAll("[a-df-zA-DF-Z]", "") // Remove most Latin letters, keep 'e' for now
            .replaceAll("\\s+", "")   // Remove extra spaces created by removals
            .trim();
    }
    
    /**
     * Check if text contains Tibetan characters
     */
    private static boolean containsTibetanChars(String text) {
        return TIBETAN_PATTERN.matcher(text).matches();
    }
    
    /**
     * Split Tibetan content into term (short) and definition (long)
     */
    private static String[] splitTibetanIntoTermAndDefinition(String tibetanContent) {
        if (tibetanContent == null || tibetanContent.isEmpty()) {
            return new String[]{"", ""};
        }
        
        // Look for natural break points to separate term from definition
        // 1. First occurrence of །followed by significant content suggests end of term
        int firstPeriod = tibetanContent.indexOf("།");
        if (firstPeriod > 0 && firstPeriod < 50) {
            // Check if there's substantial content after the period (definition)
            String afterPeriod = tibetanContent.substring(firstPeriod + 1).trim();
            if (afterPeriod.length() > 10) {
                return new String[]{
                    tibetanContent.substring(0, firstPeriod + 1),
                    afterPeriod
                };
            }
        }
        
        // 2. Look for double tsheg pattern (་་) which sometimes separates term from definition
        int doubleTsheg = tibetanContent.indexOf("་་");
        if (doubleTsheg > 10 && doubleTsheg < 50) {
            return new String[]{
                tibetanContent.substring(0, doubleTsheg),
                tibetanContent.substring(doubleTsheg + 2)
            };
        }
        
        // 3. If content is very long, split at a reasonable point (around 30-40 chars)
        if (tibetanContent.length() > 60) {
            // Look for a good split point around 25-35 character range
            for (int i = 25; i < Math.min(40, tibetanContent.length()); i++) {
                if (tibetanContent.charAt(i) == '།' || tibetanContent.charAt(i) == '་') {
                    return new String[]{
                        tibetanContent.substring(0, i + 1),
                        tibetanContent.substring(i + 1)
                    };
                }
            }
            
            // Fallback: hard split at 30 chars
            return new String[]{
                tibetanContent.substring(0, 30) + "།",
                tibetanContent.substring(30)
            };
        }
        
        // 4. Short content - treat as term only
        return new String[]{tibetanContent, ""};
    }

    /**
     * Represents a 4-column glossary entry
     */
    static class FourColumnEntry {
        final String english;
        final String subject;
        final String tibetanTerm;
        final String tibetanDefinition;

        FourColumnEntry(String english, String subject, String tibetanTerm, String tibetanDefinition) {
            this.english = english;
            this.subject = subject;
            this.tibetanTerm = tibetanTerm;
            this.tibetanDefinition = tibetanDefinition;
        }

        @Override
        public String toString() {
            return String.format("Entry[%s|%s|%s|%s...]", english, subject, tibetanTerm, 
                    tibetanDefinition.length() > 20 ? tibetanDefinition.substring(0, 20) + "..." : tibetanDefinition);
        }
    }



}