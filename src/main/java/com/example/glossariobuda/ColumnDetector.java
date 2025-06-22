package com.example.glossariobuda;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Spatial Column Detector for Buddhist Glossary OCR
 * 
 * Analyzes OCR text to detect column boundaries based on spacing patterns
 * and reconstructs the 4-column structure:
 * [English Term] [Subject] [Tibetan Term] [Tibetan Definition]
 */
public class ColumnDetector {
    
    public static class ColumnizedTerm {
        public String englishTerm;
        public String subject;
        public String tibetanTerm;
        public String tibetanDefinition;
        
        public ColumnizedTerm(String english, String subject, String tibetan, String definition) {
            this.englishTerm = english != null ? english.trim() : "";
            this.subject = subject != null ? subject.trim() : "";
            this.tibetanTerm = tibetan != null ? tibetan.trim() : "";
            this.tibetanDefinition = definition != null ? definition.trim() : "";
        }
        
        @Override
        public String toString() {
            return String.format("'%s' [%s] -> '%s' | Def: '%s'", 
                englishTerm, subject, tibetanTerm, 
                tibetanDefinition.length() > 50 ? tibetanDefinition.substring(0, 50) + "..." : tibetanDefinition);
        }
    }
    
    public static List<ColumnizedTerm> detectColumns(String ocrText) {
        List<ColumnizedTerm> terms = new ArrayList<>();
        
        // First, reconstruct broken lines to handle OCR fragmentation
        String reconstructedText = reconstructBrokenLines(ocrText);
        
        // Split text into lines for spatial analysis
        String[] lines = reconstructedText.split("\\n");
        
        for (String line : lines) {
            if (line.trim().isEmpty()) continue;
            
            // Use hybrid detection approach
            ColumnizedTerm term = detectColumnsHybrid(line);
            if (term != null) {
                terms.add(term);
                System.out.println("[COLUMN] " + term);
            }
        }
        
        return terms;
    }
    
    private static String reconstructBrokenLines(String ocrText) {
        System.out.println("[RECONSTRUCT] Starting multi-column line reconstruction...");
        System.out.println("[RECONSTRUCT] Original OCR preview: " + 
            ocrText.substring(0, Math.min(200, ocrText.length())) + "...");
        
        String[] lines = ocrText.split("\\n");
        List<ReconstructedTerm> terms = new ArrayList<>();
        ReconstructedTerm currentTerm = null;
        
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (line.isEmpty()) continue;
            
            System.out.println("[RECONSTRUCT] Processing line: " + line);
            
            // Check if this line starts with English (indicates new term)
            if (startsWithEnglish(line)) {
                // Save previous term if we have one
                if (currentTerm != null) {
                    terms.add(currentTerm);
                    System.out.println("[RECONSTRUCT] Completed term: " + currentTerm.getReconstructedLine());
                }
                
                // Start new term and analyze initial column distribution
                currentTerm = new ReconstructedTerm();
                analyzeLineIntoColumns(line, currentTerm);
                System.out.println("[RECONSTRUCT] Started new term from: " + line);
            } else {
                // This is a continuation line - distribute content to appropriate columns
                if (currentTerm != null) {
                    distributeContinuationLine(line, currentTerm);
                    System.out.println("[RECONSTRUCT] Appended continuation: " + line);
                } else {
                    // Orphaned line - create new term
                    currentTerm = new ReconstructedTerm();
                    analyzeLineIntoColumns(line, currentTerm);
                    System.out.println("[RECONSTRUCT] Started orphaned term: " + line);
                }
            }
        }
        
        // Add the last term
        if (currentTerm != null) {
            terms.add(currentTerm);
            System.out.println("[RECONSTRUCT] Final term: " + currentTerm.getReconstructedLine());
        }
        
        // Convert back to line format
        List<String> reconstructedLines = new ArrayList<>();
        for (ReconstructedTerm term : terms) {
            reconstructedLines.add(term.getReconstructedLine());
        }
        
        String result = String.join("\n", reconstructedLines);
        System.out.println("[RECONSTRUCT] Multi-column reconstruction complete. Result preview: " + 
            result.substring(0, Math.min(200, result.length())) + "...");
        
        return result;
    }
    
    private static class ReconstructedTerm {
        StringBuilder englishPart = new StringBuilder();
        StringBuilder subjectPart = new StringBuilder();
        StringBuilder tibetanTermPart = new StringBuilder();
        StringBuilder tibetanDefPart = new StringBuilder();
        
        String getReconstructedLine() {
            return englishPart.toString().trim() + " " + 
                   subjectPart.toString().trim() + " " + 
                   tibetanTermPart.toString().trim() + " " + 
                   tibetanDefPart.toString().trim();
        }
    }
    
    private static void analyzeLineIntoColumns(String line, ReconstructedTerm term) {
        // Use content-based analysis to distribute initial line into columns
        int englishEnd = findEnglishEnd(line);
        int subjectEnd = findSubjectEnd(line, englishEnd);
        int tibetanStart = findTibetanStart(line, subjectEnd);
        
        // Extract and store initial column content
        if (englishEnd > 0) {
            term.englishPart.append(line.substring(0, englishEnd).trim());
        }
        
        if (subjectEnd > englishEnd) {
            String subject = line.substring(englishEnd, subjectEnd).trim();
            if (!subject.isEmpty()) {
                term.subjectPart.append(subject);
            }
        }
        
        if (tibetanStart > 0 && tibetanStart < line.length()) {
            String tibetanContent = line.substring(tibetanStart).trim();
            // Initially put all Tibetan content in term part - will be refined later
            term.tibetanTermPart.append(tibetanContent);
        }
        
        System.out.println("[ANALYZE_COLUMNS] English: '" + term.englishPart + "' Subject: '" + 
            term.subjectPart + "' Tibetan: '" + term.tibetanTermPart + "'");
    }
    
    private static void distributeContinuationLine(String line, ReconstructedTerm term) {
        String trimmedLine = line.trim();
        
        // Determine what type of content this continuation contains
        boolean hasEnglish = trimmedLine.matches(".*[a-zA-Z].*");
        boolean hasTibetan = trimmedLine.matches(".*[\\u0F00-\\u0FFF].*");
        boolean hasSubjectCode = trimmedLine.matches(".*\\b[a-z]{2,6}\\b.*");
        
        System.out.println("[DISTRIBUTE] Line content - English: " + hasEnglish + 
            ", Tibetan: " + hasTibetan + ", Subject: " + hasSubjectCode);
        
        if (hasTibetan && !hasEnglish) {
            // Pure Tibetan content - determine if it's term or definition
            if (term.tibetanTermPart.length() > 0 && 
                (trimmedLine.contains("།") || trimmedLine.length() > 30)) {
                // Likely definition content
                if (term.tibetanDefPart.length() > 0) {
                    term.tibetanDefPart.append(" ");
                }
                term.tibetanDefPart.append(trimmedLine);
                System.out.println("[DISTRIBUTE] Added to Tibetan definition");
            } else {
                // Likely term continuation
                if (term.tibetanTermPart.length() > 0) {
                    term.tibetanTermPart.append(" ");
                }
                term.tibetanTermPart.append(trimmedLine);
                System.out.println("[DISTRIBUTE] Added to Tibetan term");
            }
        } else if (hasEnglish && !hasTibetan) {
            // English content - likely continuing English term
            if (term.englishPart.length() > 0) {
                term.englishPart.append(" ");
            }
            term.englishPart.append(trimmedLine);
            System.out.println("[DISTRIBUTE] Added to English term");
        } else if (hasSubjectCode && !hasTibetan) {
            // Subject code continuation
            if (term.subjectPart.length() > 0) {
                term.subjectPart.append(" ");
            }
            term.subjectPart.append(trimmedLine);
            System.out.println("[DISTRIBUTE] Added to subject");
        } else {
            // Mixed content or unclear - distribute based on position and content
            distributeMixedContent(trimmedLine, term);
        }
    }
    
    private static void distributeMixedContent(String line, ReconstructedTerm term) {
        // Try to split mixed content line into appropriate columns
        System.out.println("[DISTRIBUTE_MIXED] Processing mixed line: " + line);
        
        // Look for clear column boundaries in the mixed line
        String[] parts = line.split("\\s{2,}"); // Split on multiple spaces
        
        if (parts.length > 1) {
            // Multiple parts - distribute across columns
            for (int i = 0; i < parts.length; i++) {
                String part = parts[i].trim();
                
                if (part.matches(".*[a-zA-Z].*") && !part.matches(".*[\\u0F00-\\u0FFF].*")) {
                    // English part
                    if (term.englishPart.length() > 0) term.englishPart.append(" ");
                    term.englishPart.append(part);
                } else if (part.matches("^[a-z]{2,6}$")) {
                    // Subject code
                    if (term.subjectPart.length() > 0) term.subjectPart.append(" ");
                    term.subjectPart.append(part);
                } else if (part.matches(".*[\\u0F00-\\u0FFF].*")) {
                    // Tibetan content
                    if (term.tibetanTermPart.length() > 0) term.tibetanTermPart.append(" ");
                    term.tibetanTermPart.append(part);
                }
            }
        } else {
            // Single part - add to most appropriate column based on content
            if (line.matches(".*[\\u0F00-\\u0FFF].*")) {
                // Has Tibetan - add to Tibetan content
                if (term.tibetanTermPart.length() > 0) term.tibetanTermPart.append(" ");
                term.tibetanTermPart.append(line);
            } else {
                // No Tibetan - likely English continuation
                if (term.englishPart.length() > 0) term.englishPart.append(" ");
                term.englishPart.append(line);
            }
        }
        
        System.out.println("[DISTRIBUTE_MIXED] Result - English: '" + term.englishPart + 
            "' Subject: '" + term.subjectPart + "' Tibetan: '" + term.tibetanTermPart + "'");
    }
    
    
    private static boolean startsWithEnglish(String line) {
        if (line.isEmpty()) return false;
        char first = line.charAt(0);
        return (first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z');
    }
    
    private static ColumnizedTerm analyzeLineColumns(String line) {
        // Clean the line
        String cleaned = line.trim();
        if (cleaned.length() < 10) return null;
        
        System.out.println("[ANALYZE] Line: " + cleaned);
        
        // Method 1: Detect columns by large spacing gaps (3+ spaces)
        ColumnizedTerm result = detectBySpacing(cleaned);
        if (result != null) return result;
        
        // Method 2: Detect by pattern matching with known structure
        result = detectByPattern(cleaned);
        if (result != null) return result;
        
        // Method 3: Detect by character type transitions (English -> Tibetan)
        result = detectByCharacterTransition(cleaned);
        if (result != null) return result;
        
        return null;
    }
    private static ColumnizedTerm detectBySpacing(String line) {
        // Debug: Count whitespace patterns in the line
        debugWhitespacePatterns(line);
        
        // Analyze all space gaps and find the most significant column boundaries
        String[] parts = splitByColumnGaps(line);
        System.out.println("[SPACING] Column gap analysis found " + parts.length + " parts");
        
        if (parts.length < 3) {
            // Fallback: Use intelligent splitting based on content
            parts = intelligentSplit(line);
            System.out.println("[SPACING] Fallback: Found " + parts.length + " parts");
        }

        if (parts.length >= 3) {
            System.out.println("[SPACING] Using " + parts.length + " parts for column detection");
            for (int i = 0; i < parts.length; i++) {
                System.out.println("  Part " + i + ": '" + parts[i] + "'");
            }

            String english = parts[0].trim();
            String subject = "";
            String tibetanTerm = "";
            String tibetanDef = "";

            // Analyze the parts - simple column assignment
            if (parts.length >= 4) {
                // 4 clear columns
                subject = parts[1].trim();
                tibetanTerm = parts[2].trim();
                tibetanDef = parts[3].trim();

            } else if (parts.length == 3) {
                // Need to split the last part into term and definition
                if (isSubjectCode(parts[1])) {
                    subject = parts[1].trim();
                    // parts[2] contains both term and definition - split generically
                    String allContent = parts[2].trim();
                    String[] contentParts = splitContentIntoTermAndDefinition(allContent);
                    tibetanTerm = contentParts[0];
                    tibetanDef = contentParts[1];
                } else {
                    // parts[1] might be mixed English + subject
                    String[] englishSubject = splitEnglishAndSubject(parts[1]);
                    if (!englishSubject[1].isEmpty()) {
                        english = (english + " " + englishSubject[0]).trim();
                        subject = englishSubject[1];
                        String[] contentParts = splitContentIntoTermAndDefinition(parts[2]);
                        tibetanTerm = contentParts[0];
                        tibetanDef = contentParts[1];
                    }
                }
            }

            if (isValidTerm(english) && hasNonLatinContent(tibetanTerm)) {
                return new ColumnizedTerm(english, subject, tibetanTerm, tibetanDef);
            }
        }

        return null;
    }
    
    private static String[] splitByColumnGaps(String line) {
        System.out.println("[COLUMN_GAPS] Analyzing line for column boundaries: " + line);
        
        // Find all space sequences and their positions
        List<SpaceGap> gaps = new ArrayList<>();
        java.util.regex.Pattern spacePattern = java.util.regex.Pattern.compile("\\s+");
        java.util.regex.Matcher matcher = spacePattern.matcher(line);
        
        while (matcher.find()) {
            gaps.add(new SpaceGap(matcher.start(), matcher.end(), matcher.group().length()));
        }
        
        System.out.println("[COLUMN_GAPS] Found " + gaps.size() + " space gaps");
        
        if (gaps.size() < 2) {
            // Not enough gaps for columns
            return new String[]{line};
        }
        
        // Find the most significant gaps for column separation
        // Strategy: Look for gaps that are significantly larger than average
        gaps.sort((a, b) -> Integer.compare(b.length, a.length)); // Sort by length descending
        
        List<SpaceGap> columnSeparators = new ArrayList<>();
        
        // Take the largest gaps as potential column separators
        int maxSeparators = Math.min(3, gaps.size()); // Maximum 3 separators for 4 columns
        for (int i = 0; i < maxSeparators; i++) {
            SpaceGap gap = gaps.get(i);
            System.out.println("[COLUMN_GAPS] Considering gap at position " + gap.start + 
                " (length " + gap.length + ") as column separator");
            columnSeparators.add(gap);
        }
        
        // Sort separators by position
        columnSeparators.sort((a, b) -> Integer.compare(a.start, b.start));
        
        // Split the line at these positions
        List<String> parts = new ArrayList<>();
        int lastEnd = 0;
        
        for (SpaceGap separator : columnSeparators) {
            if (separator.start > lastEnd) {
                String part = line.substring(lastEnd, separator.start).trim();
                if (!part.isEmpty()) {
                    parts.add(part);
                    System.out.println("[COLUMN_GAPS] Column part: '" + part + "'");
                }
            }
            lastEnd = separator.end;
        }
        
        // Add the final part
        if (lastEnd < line.length()) {
            String finalPart = line.substring(lastEnd).trim();
            if (!finalPart.isEmpty()) {
                parts.add(finalPart);
                System.out.println("[COLUMN_GAPS] Final column part: '" + finalPart + "'");
            }
        }
        
        return parts.toArray(new String[0]);
    }
    
    private static class SpaceGap {
        int start;
        int end;
        int length;
        
        SpaceGap(int start, int end, int length) {
            this.start = start;
            this.end = end;
            this.length = length;
        }
    }
    
    private static void debugWhitespacePatterns(String line) {
        System.out.println("[WHITESPACE] Analyzing line: " + line);
        
        // Count different whitespace patterns
        int singleSpaces = 0;
        int doubleSpaces = 0;
        int tripleSpaces = 0;
        int quadSpaces = 0;
        int longerSpaces = 0;
        
        // Find all whitespace sequences
        java.util.regex.Pattern spacePattern = java.util.regex.Pattern.compile("\\s+");
        java.util.regex.Matcher matcher = spacePattern.matcher(line);
        
        while (matcher.find()) {
            String spaceSequence = matcher.group();
            int spaceCount = spaceSequence.length();
            
            if (spaceCount == 1) singleSpaces++;
            else if (spaceCount == 2) doubleSpaces++;
            else if (spaceCount == 3) tripleSpaces++;
            else if (spaceCount == 4) quadSpaces++;
            else longerSpaces++;
            
            System.out.println("[WHITESPACE]   Found " + spaceCount + " spaces at position " + matcher.start());
        }
        
        System.out.println("[WHITESPACE] Summary: 1-space=" + singleSpaces + ", 2-space=" + doubleSpaces + 
                          ", 3-space=" + tripleSpaces + ", 4-space=" + quadSpaces + ", 5+-space=" + longerSpaces);
        
        // Visual representation
        StringBuilder visual = new StringBuilder();
        for (char c : line.toCharArray()) {
            if (c == ' ') {
                visual.append("␣");
            } else {
                visual.append(c);
            }
        }
        System.out.println("[WHITESPACE] Visual: " + visual.toString());
    }
    
    private static String[] intelligentSplit(String line) {
        System.out.println("[INTELLIGENT] Analyzing line: " + line);
        
        // Look for pattern: English words + subject code + Tibetan text
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s-']+?)\\s+" +           // English term
            "([a-z]{2,6}(?:,[a-z]{2,6})*)\\s+" +         // Subject code(s)
            "([\\u0F00-\\u0FFF].*)$"                     // Tibetan content
        );
        
        java.util.regex.Matcher matcher = pattern.matcher(line);
        if (matcher.find()) {
            String english = matcher.group(1).trim();
            String subject = matcher.group(2).trim();
            String tibetan = matcher.group(3).trim();
            
            System.out.println("[INTELLIGENT] Pattern match: '" + english + "' | '" + subject + "' | '" + tibetan.substring(0, Math.min(30, tibetan.length())) + "...'");
            return new String[]{english, subject, tibetan};
        }
        
        // Fallback: try to find English -> Tibetan transition
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c >= '\u0F00' && c <= '\u0FFF') {
                // Found first Tibetan character
                String beforeTibetan = line.substring(0, i).trim();
                String tibetanPart = line.substring(i).trim();
                
                // Try to split the English part into term + subject
                String[] englishParts = beforeTibetan.split("\\s+");
                if (englishParts.length >= 2) {
                    String lastPart = englishParts[englishParts.length - 1];
                    
                    // Check if last part looks like subject code
                    if (lastPart.matches("^[a-z]{2,6}(?:,[a-z]{2,6})*$")) {
                        String english = String.join(" ", java.util.Arrays.copyOfRange(englishParts, 0, englishParts.length - 1));
                        String subject = lastPart;
                        
                        System.out.println("[INTELLIGENT] English/Tibetan split: '" + english + "' | '" + subject + "' | '" + tibetanPart.substring(0, Math.min(30, tibetanPart.length())) + "...'");
                        return new String[]{english, subject, tibetanPart};
                    }
                }
                
                // No clear subject, treat all before Tibetan as English
                System.out.println("[INTELLIGENT] Simple English/Tibetan split: '" + beforeTibetan + "' | '' | '" + tibetanPart.substring(0, Math.min(30, tibetanPart.length())) + "...'");
                return new String[]{beforeTibetan, "", tibetanPart};
            }
        }
        
        // No Tibetan found, return original as single part
        System.out.println("[INTELLIGENT] No clear structure found");
        return new String[]{line};
    }

    
    private static ColumnizedTerm detectByPattern(String line) {
        // Pattern: English words + subject code + content
        Pattern pattern = Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s-']+?)\\s+" +           // English term
            "([a-z]{2,6})\\s+" +                         // Subject code
            "(.+)$"                                      // All remaining content
        );
        
        Matcher matcher = pattern.matcher(line);
        if (matcher.find()) {
            String english = matcher.group(1).trim();
            String subject = matcher.group(2).trim();
            String allContent = matcher.group(3).trim();
            
            String[] contentParts = splitContentIntoTermAndDefinition(allContent);
            String term = contentParts[0];
            String definition = contentParts[1];
            
            System.out.println("[PATTERN] English: '" + english + "' Subject: '" + subject + "' Content: '" + 
                allContent.substring(0, Math.min(30, allContent.length())) + "...'");
            
            return new ColumnizedTerm(english, subject, term, definition);
        }
        
        return null;
    }
    
    private static ColumnizedTerm detectByCharacterTransition(String line) {
        // Find the transition from English to non-Latin characters
        int nonLatinStart = -1;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            // Look for non-Latin scripts (Tibetan, Chinese, etc.)
            if (c >= '\u0F00' && c <= '\u0FFF' || // Tibetan
                c >= '\u4E00' && c <= '\u9FFF' || // CJK
                c >= '\u0590' && c <= '\u05FF') { // Hebrew/Arabic etc.
                nonLatinStart = i;
                break;
            }
        }
        
        if (nonLatinStart > 0) {
            String englishPart = line.substring(0, nonLatinStart).trim();
            String contentPart = line.substring(nonLatinStart).trim();
            
            // Extract subject from English part
            String[] englishSubject = splitEnglishAndSubject(englishPart);
            String english = englishSubject[0];
            String subject = englishSubject[1];
            
            // Split content part generically
            String[] contentParts = splitContentIntoTermAndDefinition(contentPart);
            String term = contentParts[0];
            String definition = contentParts[1];
            
            if (isValidTerm(english)) {
                System.out.println("[TRANSITION] English: '" + english + "' Subject: '" + subject + "'");
                return new ColumnizedTerm(english, subject, term, definition);
            }
        }
        
        return null;
    }
    
    private static String[] splitEnglishAndSubject(String text) {
        // Look for subject code at the end
        Pattern pattern = Pattern.compile("^(.+?)\\s+([a-z]{2,6})$");
        Matcher matcher = pattern.matcher(text.trim());
        
        if (matcher.find()) {
            return new String[]{matcher.group(1).trim(), matcher.group(2).trim()};
        }
        
        return new String[]{text.trim(), ""};
    }
    
    // Generic method to split content into term and definition
    private static String[] splitContentIntoTermAndDefinition(String content) {
        String cleaned = content.trim();
        System.out.println("[SPLIT] Processing content: " + cleaned.substring(0, Math.min(50, cleaned.length())) + "...");
        
        // Method 1: Split on punctuation marks (། for Tibetan, . for others)
        int firstPunctuation = -1;
        char[] punctuationMarks = {'།', '.', '!', '?', ';'};
        
        for (char punct : punctuationMarks) {
            int index = cleaned.indexOf(punct);
            if (index > 0 && index < cleaned.length() - 1) {
                if (firstPunctuation == -1 || index < firstPunctuation) {
                    firstPunctuation = index;
                }
            }
        }
        
        if (firstPunctuation > 0) {
            String term = cleaned.substring(0, firstPunctuation + 1).trim();
            String definition = cleaned.substring(firstPunctuation + 1).trim();
            
            // Validate term length is reasonable
            if (term.length() >= 3 && term.length() <= 50 && definition.length() > 3) {
                System.out.println("[SPLIT] Split on punctuation: '" + term + "' | '" + 
                    definition.substring(0, Math.min(30, definition.length())) + "...'");
                return new String[]{term, definition};
            }
        }
        
        // Method 2: Split on double or triple spaces
        String[] spaceParts = cleaned.split("\\s{2,}");
        if (spaceParts.length >= 2) {
            String term = spaceParts[0].trim();
            String definition = String.join(" ", java.util.Arrays.copyOfRange(spaceParts, 1, spaceParts.length));
            System.out.println("[SPLIT] Split on spaces: '" + term + "' | '" + 
                definition.substring(0, Math.min(30, definition.length())) + "...'");
            return new String[]{term, definition};
        }
        
        // Method 3: Split on multiple spaces (if column data got merged)
        String[] multiSpaceParts = cleaned.split("\\s{2,}");
        if (multiSpaceParts.length >= 2) {
            String term = multiSpaceParts[0].trim();
            String definition = String.join(" ", java.util.Arrays.copyOfRange(multiSpaceParts, 1, multiSpaceParts.length));
            System.out.println("[SPLIT] Multiple space split: '" + term + "' | '" + 
                definition.substring(0, Math.min(30, definition.length())) + "...'");
            return new String[]{term, definition};
        }
        
        // Method 4: Fallback - take first few words as term
        String[] words = cleaned.split("\\s+");
        if (words.length > 4) {
            String term = String.join(" ", java.util.Arrays.copyOfRange(words, 0, Math.min(3, words.length)));
            String definition = String.join(" ", java.util.Arrays.copyOfRange(words, 3, words.length));
            System.out.println("[SPLIT] Fallback split: '" + term + "' | '" + 
                definition.substring(0, Math.min(30, definition.length())) + "...'");
            return new String[]{term, definition};
        }
        
        // No split possible - return as single term
        System.out.println("[SPLIT] No split possible, returning as single term");
        return new String[]{cleaned, ""};
    }
    
    private static boolean isSubjectCode(String text) {
        return text != null && text.matches("^[a-z]{2,6}$");
    }
    
    private static boolean isValidTerm(String text) {
        return text != null && text.length() >= 3 && text.matches(".*[a-zA-Z].*");
    }
    
    private static boolean hasNonLatinContent(String text) {
        if (text == null || text.trim().isEmpty()) return false;
        for (char c : text.toCharArray()) {
            // Check for various non-Latin scripts
            if (c >= '\u0F00' && c <= '\u0FFF' || // Tibetan
                c >= '\u4E00' && c <= '\u9FFF' || // CJK
                c >= '\u0590' && c <= '\u05FF' || // Hebrew/Arabic
                c >= '\u0400' && c <= '\u04FF') { // Cyrillic
                return true;
            }
        }
        return false;
    }
    
    /**
     * Hybrid Column Detection combining fixed-width templates with content-based boundaries
     * 
     * Expected 4-column structure:
     * [English Term] | [Subject Code] | [Tibetan Term] | [Tibetan Definition]
     * 
     * Strategy:
     * 1. Use fixed-width column templates as guides
     * 2. Find content-based boundaries using character transitions
     * 3. Reconstruct Tibetan content properly across term/definition split
     */
    private static ColumnizedTerm detectColumnsHybrid(String line) {
        System.out.println("[HYBRID] Processing line: " + line);
        
        String cleaned = line.trim();
        if (cleaned.length() < 10) return null;
        
        // Method 1: Smart content-based detection using word boundaries
        ColumnizedTerm smartResult = detectBySmartBoundaries(cleaned);
        if (smartResult != null && isValidColumnTerm(smartResult)) {
            System.out.println("[HYBRID] Smart boundary detection successful");
            return smartResult;
        }
        
        // Method 2: Pattern-based detection for known formats
        ColumnizedTerm patternResult = detectByAdvancedPattern(cleaned);
        if (patternResult != null && isValidColumnTerm(patternResult)) {
            System.out.println("[HYBRID] Pattern detection successful");
            return patternResult;
        }
        
        // Method 3: Fallback to content boundaries
        ColumnizedTerm contentResult = detectByContentBoundaries(cleaned);
        if (contentResult != null && isValidColumnTerm(contentResult)) {
            System.out.println("[HYBRID] Content-based detection successful");
            return contentResult;
        }
        
        return null;
    }
    
    /**
     * Smart boundary detection that respects word boundaries and content types
     * Analyzes actual word patterns instead of fixed character positions
     */
    private static ColumnizedTerm detectBySmartBoundaries(String line) {
        System.out.println("[SMART] Analyzing line with smart boundaries: " + line);
        
        // Split into words for analysis
        String[] words = line.split("\\s+");
        System.out.println("[SMART] Found " + words.length + " words");
        
        // Find column boundaries by analyzing word types
        int englishEnd = -1;
        int subjectEnd = -1;
        int tibetanStart = -1;
        
        // Phase 1: Find English section (consecutive Latin words)
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            if (isEnglishWord(word)) {
                englishEnd = i;
            } else {
                break; // First non-English word
            }
        }
        
        // Phase 2: Check for subject code after English
        if (englishEnd >= 0 && englishEnd + 1 < words.length) {
            String candidateSubject = words[englishEnd + 1];
            if (isSubjectCode(candidateSubject)) {
                subjectEnd = englishEnd + 1;
                System.out.println("[SMART] Found subject code: '" + candidateSubject + "'");
            } else {
                // Subject code might be missing or combined with English
                subjectEnd = englishEnd;
            }
        } else {
            subjectEnd = englishEnd;
        }
        
        // Phase 3: Find start of Tibetan content
        for (int i = Math.max(0, subjectEnd + 1); i < words.length; i++) {
            String word = words[i];
            if (containsTibetanChars(word)) {
                tibetanStart = i;
                System.out.println("[SMART] Found Tibetan start at word " + i + ": '" + word + "'");
                break;
            }
        }
        
        // Validate we found reasonable boundaries
        if (englishEnd < 0 || tibetanStart < 0) {
            System.out.println("[SMART] Could not find valid boundaries");
            return null;
        }
        
        // Extract sections
        String english = String.join(" ", java.util.Arrays.copyOfRange(words, 0, englishEnd + 1));
        String subject = "";
        if (subjectEnd > englishEnd) {
            subject = words[subjectEnd];
        }
        String allTibetan = String.join(" ", java.util.Arrays.copyOfRange(words, tibetanStart, words.length));
        
        // Clean up OCR noise from English
        english = cleanEnglishTerm(english);
        
        // Split Tibetan content into term and definition
        String[] tibetanParts = smartSplitTibetan(allTibetan);
        String tibetanTerm = tibetanParts[0];
        String tibetanDef = tibetanParts[1];
        
        System.out.println("[SMART] Result - English: '" + english + "' Subject: '" + subject + 
            "' Term: '" + tibetanTerm + "' Def: '" + tibetanDef.substring(0, Math.min(30, tibetanDef.length())) + "...'");
        
        return new ColumnizedTerm(english, subject, tibetanTerm, tibetanDef);
    }
    
    /**
     * Advanced pattern detection for known Buddhist glossary formats
     */
    private static ColumnizedTerm detectByAdvancedPattern(String line) {
        System.out.println("[PATTERN] Advanced pattern matching: " + line);
        
        // Pattern 1: English + subject + Tibetan (most common)
        java.util.regex.Pattern pattern1 = java.util.regex.Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s'-]+?)\\s+([a-z]{2,6})\\s+([\\u0F00-\\u0FFF][\\s\\u0F00-\\u0FFF\\u0020-\\u007F]*)$"
        );
        
        java.util.regex.Matcher matcher1 = pattern1.matcher(line);
        if (matcher1.find()) {
            String english = cleanEnglishTerm(matcher1.group(1));
            String subject = matcher1.group(2);
            String allTibetan = matcher1.group(3).trim();
            
            String[] tibetanParts = smartSplitTibetan(allTibetan);
            System.out.println("[PATTERN] Pattern 1 matched - English: '" + english + "' Subject: '" + subject + "'");
            return new ColumnizedTerm(english, subject, tibetanParts[0], tibetanParts[1]);
        }
        
        // Pattern 2: English + Tibetan (no subject code)
        java.util.regex.Pattern pattern2 = java.util.regex.Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s'-]+?)\\s+([\\u0F00-\\u0FFF][\\s\\u0F00-\\u0FFF\\u0020-\\u007F]*)$"
        );
        
        java.util.regex.Matcher matcher2 = pattern2.matcher(line);
        if (matcher2.find()) {
            String english = cleanEnglishTerm(matcher2.group(1));
            String allTibetan = matcher2.group(2).trim();
            
            String[] tibetanParts = smartSplitTibetan(allTibetan);
            System.out.println("[PATTERN] Pattern 2 matched - English: '" + english + "' (no subject)");
            return new ColumnizedTerm(english, "", tibetanParts[0], tibetanParts[1]);
        }
        
        return null;
    }
    
    private static boolean isEnglishWord(String word) {
        // Check if word is primarily English (Latin characters)
        if (word == null || word.trim().isEmpty()) return false;
        
        // Remove common OCR noise and punctuation
        String cleaned = word.replaceAll("[^a-zA-Z'-]", "");
        if (cleaned.length() < 2) return false;
        
        // Must be primarily Latin characters
        int latinCount = 0;
        for (char c : cleaned.toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '\'' || c == '-') {
                latinCount++;
            }
        }
        
        return latinCount >= cleaned.length() * 0.8; // At least 80% Latin
    }
    
    private static boolean containsTibetanChars(String word) {
        if (word == null) return false;
        for (char c : word.toCharArray()) {
            if (c >= '\u0F00' && c <= '\u0FFF') {
                return true;
            }
        }
        return false;
    }
    
    private static String cleanEnglishTerm(String english) {
        if (english == null) return "";
        
        // Remove common OCR noise
        String cleaned = english
            .replaceAll("\\s+x\\s*$", "") // Remove trailing "x"
            .replaceAll("\\s+[0-9]+\\s*$", "") // Remove trailing numbers
            .replaceAll("[^a-zA-Z\\s'-]", " ") // Remove non-English chars except apostrophes and hyphens
            .replaceAll("\\s+", " ") // Normalize spaces
            .trim();
        
        System.out.println("[CLEAN] English: '" + english + "' -> '" + cleaned + "'");
        return cleaned;
    }
    
    /**
     * Smart Tibetan splitting that properly identifies term vs definition boundaries
     */
    private static String[] smartSplitTibetan(String tibetanText) {
        String cleaned = tibetanText.trim();
        System.out.println("[SMART_TIBETAN] Processing: " + cleaned.substring(0, Math.min(50, cleaned.length())) + "...");
        
        // Method 1: Look for རྐྱེན། completion - this is the key marker
        String[] rkyenResult = findRkyenCompletion(cleaned);
        if (!rkyenResult[0].isEmpty()) {
            System.out.println("[SMART_TIBETAN] Found རྐྱེན། completion");
            return rkyenResult;
        }
        
        // Method 2: Look for other common Tibetan term endings
        String[] endingResult = findTibetanTermEnding(cleaned);
        if (!endingResult[0].isEmpty()) {
            System.out.println("[SMART_TIBETAN] Found term ending marker");
            return endingResult;
        }
        
        // Method 3: Split on significant punctuation
        String[] punctResult = findPunctuationSplit(cleaned);
        if (!punctResult[0].isEmpty()) {
            System.out.println("[SMART_TIBETAN] Used punctuation split");
            return punctResult;
        }
        
        // Method 4: Intelligent word-count based split
        String[] words = cleaned.split("\\s+");
        if (words.length > 6) {
            // Terms are typically 1-4 words, definitions are longer
            int splitPoint = Math.min(4, words.length / 3);
            String term = String.join(" ", java.util.Arrays.copyOfRange(words, 0, splitPoint));
            String definition = String.join(" ", java.util.Arrays.copyOfRange(words, splitPoint, words.length));
            
            System.out.println("[SMART_TIBETAN] Used intelligent word split at position " + splitPoint);
            return new String[]{term, definition};
        }
        
        // Fallback: return as single term
        System.out.println("[SMART_TIBETAN] No clear split found, returning as term");
        return new String[]{cleaned, ""};
    }
    
    private static String[] findRkyenCompletion(String text) {
        int rkyenIndex = text.indexOf("རྐྱེན།");
        if (rkyenIndex > 0) {
            // Find the word that should end with རྐྱེན།
            String beforeRkyen = text.substring(0, rkyenIndex);
            String afterRkyen = text.substring(rkyenIndex + 5).trim();
            
            // Look for completion patterns like པའི་, བའི་, མིན་པའི་
            String[] words = beforeRkyen.split("\\s+");
            for (int i = words.length - 1; i >= 0; i--) {
                String word = words[i];
                if (word.matches(".*[པབ]འ[ིེ]་$") || word.contains("མིན་པ")) {
                    // Found the completing word
                    String completeTerm = word + "རྐྱེན།";
                    
                    // Build definition from remaining parts
                    String definition = "";
                    if (i > 0) {
                        definition = String.join(" ", java.util.Arrays.copyOfRange(words, 0, i)) + " ";
                    }
                    definition += afterRkyen;
                    
                    return new String[]{completeTerm, definition.trim()};
                }
            }
        }
        return new String[]{"", ""};
    }
    
    private static String[] findTibetanTermEnding(String text) {
        // Look for other common term endings
        String[] termEndings = {"།", "༎", "༏", "ལ་གོ།", "ཞེས་བྱ།"};
        
        for (String ending : termEndings) {
            int index = text.indexOf(ending);
            if (index > 5 && index < text.length() - 5) {
                String term = text.substring(0, index + ending.length()).trim();
                String definition = text.substring(index + ending.length()).trim();
                
                // Validate reasonable lengths
                if (term.length() >= 3 && term.length() <= 50 && definition.length() >= 3) {
                    return new String[]{term, definition};
                }
            }
        }
        return new String[]{"", ""};
    }
    
    private static String[] findPunctuationSplit(String text) {
        // Find the first significant punctuation that could separate term from definition
        char[] punctMarks = {'།', '༏', '༎', '༑'};
        
        for (char punct : punctMarks) {
            int index = text.indexOf(punct);
            if (index > 8 && index < text.length() - 8) { // Ensure reasonable term and definition lengths
                String term = text.substring(0, index + 1).trim();
                String definition = text.substring(index + 1).trim();
                
                if (term.length() >= 3 && definition.length() >= 3) {
                    return new String[]{term, definition};
                }
            }
        }
        return new String[]{"", ""};
    }
    
    private static ColumnizedTerm detectByFixedWidth(String line, int[] widths) {
        System.out.println("[FIXED_WIDTH] Applying fixed-width template");
        
        int currentPos = 0;
        String[] columns = new String[4];
        
        for (int i = 0; i < widths.length && currentPos < line.length(); i++) {
            int width = widths[i];
            
            if (width == -1) {
                // Last column takes the rest
                columns[i] = line.substring(currentPos).trim();
            } else {
                int endPos = Math.min(currentPos + width, line.length());
                columns[i] = line.substring(currentPos, endPos).trim();
                currentPos = endPos;
                
                // Skip whitespace between columns
                while (currentPos < line.length() && line.charAt(currentPos) == ' ') {
                    currentPos++;
                }
            }
            
            System.out.println("[FIXED_WIDTH] Column " + i + ": '" + columns[i] + "'");
        }
        
        // Validate and clean up columns
        String english = columns[0] != null ? columns[0] : "";
        String subject = columns[1] != null ? columns[1] : "";
        String tibetanTerm = columns[2] != null ? columns[2] : "";
        String tibetanDef = columns[3] != null ? columns[3] : "";
        
        // Apply content-based refinement to improve boundaries for all columns
        if (!english.isEmpty() && hasNonLatinContent(tibetanTerm + " " + tibetanDef)) {
            // Reconstruct all column content properly
            String[] finalColumns = reconstructAllColumns(english, subject, tibetanTerm, tibetanDef);
            
            return new ColumnizedTerm(finalColumns[0], finalColumns[1], finalColumns[2], finalColumns[3]);
        }
        
        return null;
    }
    
    private static ColumnizedTerm detectByContentBoundaries(String line) {
        System.out.println("[CONTENT_BOUNDARIES] Analyzing content transitions");
        
        // Find key transition points
        int englishEnd = findEnglishEnd(line);
        int subjectEnd = findSubjectEnd(line, englishEnd);
        int tibetanStart = findTibetanStart(line, subjectEnd);
        
        if (englishEnd <= 0 || tibetanStart <= 0) {
            System.out.println("[CONTENT_BOUNDARIES] Could not find clear boundaries");
            return null;
        }
        
        String english = line.substring(0, englishEnd).trim();
        String subject = "";
        
        if (subjectEnd > englishEnd) {
            subject = line.substring(englishEnd, subjectEnd).trim();
        }
        
        String allTibetan = line.substring(tibetanStart).trim();
        
        // Reconstruct Tibetan content
        String[] tibetanParts = reconstructTibetanContent(allTibetan);
        String tibetanTerm = tibetanParts[0];
        String tibetanDef = tibetanParts[1];
        
        System.out.println("[CONTENT_BOUNDARIES] English: '" + english + "' Subject: '" + subject + "'");
        System.out.println("[CONTENT_BOUNDARIES] Tibetan Term: '" + tibetanTerm + "'");
        System.out.println("[CONTENT_BOUNDARIES] Tibetan Def: '" + tibetanDef.substring(0, Math.min(30, tibetanDef.length())) + "...'");
        
        return new ColumnizedTerm(english, subject, tibetanTerm, tibetanDef);
    }
    
    private static int findEnglishEnd(String line) {
        // Find the end of English content (usually before subject code or Tibetan)
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            
            // Look for transition to subject code pattern
            if (i > 5) { // Must have some English content
                String remaining = line.substring(i).trim();
                if (remaining.matches("^[a-z]{2,6}\\s+.*") || // Subject code pattern
                    c >= '\u0F00' && c <= '\u0FFF') { // Tibetan character
                    return i;
                }
            }
        }
        
        // Fallback: find first space after reasonable English length
        for (int i = 10; i < line.length(); i++) {
            if (line.charAt(i) == ' ') {
                return i;
            }
        }
        
        return line.length() / 3; // Fallback position
    }
    
    private static int findSubjectEnd(String line, int start) {
        if (start >= line.length()) return start;
        
        // Skip spaces
        while (start < line.length() && line.charAt(start) == ' ') start++;
        
        // Look for subject code pattern (2-6 lowercase letters)
        int pos = start;
        while (pos < line.length() && 
               ((line.charAt(pos) >= 'a' && line.charAt(pos) <= 'z') || line.charAt(pos) == ',')) {
            pos++;
        }
        
        // Validate it looks like a subject code
        String candidate = line.substring(start, pos).trim();
        if (candidate.matches("^[a-z]{2,6}(?:,[a-z]{2,6})*$")) {
            return pos;
        }
        
        return start; // No subject code found
    }
    
    private static int findTibetanStart(String line, int start) {
        // Find first Tibetan character
        for (int i = start; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c >= '\u0F00' && c <= '\u0FFF') {
                // Back up to start of word
                while (i > start && line.charAt(i - 1) != ' ') {
                    i--;
                }
                return i;
            }
        }
        
        return start; // Fallback
    }
    
    /**
     * Reconstruct Tibetan content by properly separating term from definition
     * Handles རྐྱེན། completion patterns and other Tibetan linguistic structures
     */
    private static String[] reconstructTibetanContent(String tibetanText) {
        String cleaned = tibetanText.trim();
        System.out.println("[RECONSTRUCT_TIBETAN] Processing: " + cleaned.substring(0, Math.min(50, cleaned.length())) + "...");
        
        // Method 1: Look for རྐྱེན། completion pattern
        String[] rkyenResult = splitOnRkyenPattern(cleaned);
        if (rkyenResult[0].length() > 0) {
            System.out.println("[RECONSTRUCT_TIBETAN] Used རྐྱེན། pattern");
            return rkyenResult;
        }
        
        // Method 2: Split on punctuation (། or other markers)
        String[] punctResult = splitOnTibetanPunctuation(cleaned);
        if (punctResult[0].length() > 0) {
            System.out.println("[RECONSTRUCT_TIBETAN] Used punctuation split");
            return punctResult;
        }
        
        // Method 3: Split on significant spacing
        String[] spaceResult = splitOnSignificantSpacing(cleaned);
        if (spaceResult[0].length() > 0) {
            System.out.println("[RECONSTRUCT_TIBETAN] Used spacing split");
            return spaceResult;
        }
        
        // Method 4: Fallback - split based on word count
        String[] words = cleaned.split("\\s+");
        if (words.length > 4) {
            int splitPoint = Math.min(3, words.length / 3);
            String term = String.join(" ", java.util.Arrays.copyOfRange(words, 0, splitPoint));
            String definition = String.join(" ", java.util.Arrays.copyOfRange(words, splitPoint, words.length));
            
            System.out.println("[RECONSTRUCT_TIBETAN] Used fallback word split");
            return new String[]{term, definition};
        }
        
        // No split possible
        System.out.println("[RECONSTRUCT_TIBETAN] No split possible, returning as single term");
        return new String[]{cleaned, ""};
    }
    
    private static String[] splitOnRkyenPattern(String text) {
        int rkyenIndex = text.indexOf("རྐྱེན།");
        if (rkyenIndex >= 0) {
            // Find the complete term that ends with རྐྱེན།
            String beforeRkyen = text.substring(0, rkyenIndex);
            String afterRkyen = text.substring(rkyenIndex + 5).trim();
            
            // Look for the word that should complete with རྐྱེན།
            String[] beforeWords = beforeRkyen.trim().split("\\s+");
            for (int i = beforeWords.length - 1; i >= 0; i--) {
                String word = beforeWords[i];
                if (word.matches(".*[པབ]འ[ིེ]་$") || word.contains("མིན་པའ")) {
                    // This word + རྐྱེན། forms the complete term
                    String completeTerm = word + "རྐྱེན།";
                    
                    // Build definition from remaining parts
                    String definition = "";
                    if (i > 0) {
                        definition = String.join(" ", java.util.Arrays.copyOfRange(beforeWords, 0, i)) + " ";
                    }
                    definition += afterRkyen;
                    
                    return new String[]{completeTerm, definition.trim()};
                }
            }
            
            // Fallback: last word + རྐྱེན།
            if (beforeWords.length > 0) {
                String lastWord = beforeWords[beforeWords.length - 1];
                String completeTerm = lastWord + "རྐྱེན།";
                String definition = "";
                if (beforeWords.length > 1) {
                    definition = String.join(" ", java.util.Arrays.copyOfRange(beforeWords, 0, beforeWords.length - 1)) + " ";
                }
                definition += afterRkyen;
                
                return new String[]{completeTerm, definition.trim()};
            }
        }
        
        return new String[]{"", ""};
    }
    
    private static String[] splitOnTibetanPunctuation(String text) {
        // Find first meaningful punctuation
        char[] punctMarks = {'།', '༏', '༎', '༑'};
        
        for (char punct : punctMarks) {
            int index = text.indexOf(punct);
            if (index > 5 && index < text.length() - 5) { // Reasonable position
                String term = text.substring(0, index + 1).trim();
                String definition = text.substring(index + 1).trim();
                
                // Validate split quality
                if (term.length() >= 3 && term.length() <= 50 && definition.length() >= 3) {
                    return new String[]{term, definition};
                }
            }
        }
        
        return new String[]{"", ""};
    }
    
    private static String[] splitOnSignificantSpacing(String text) {
        // Look for sequences of 2+ spaces
        String[] parts = text.split("\\s{2,}");
        if (parts.length >= 2) {
            String term = parts[0].trim();
            String definition = String.join(" ", java.util.Arrays.copyOfRange(parts, 1, parts.length));
            
            if (term.length() >= 3 && definition.length() >= 3) {
                return new String[]{term, definition};
            }
        }
        
        return new String[]{"", ""};
    }
    
    /**
     * Reconstruct all columns by cleaning and properly distributing content
     * Handles cases where content from one column might have leaked into another
     */
    private static String[] reconstructAllColumns(String english, String subject, String tibetanTerm, String tibetanDef) {
        System.out.println("[RECONSTRUCT_ALL] Input - English: '" + english + "' Subject: '" + subject + 
            "' Term: '" + tibetanTerm + "' Def: '" + tibetanDef + "'");
        
        // Clean and redistribute content
        StringBuilder cleanEnglish = new StringBuilder();
        StringBuilder cleanSubject = new StringBuilder();
        StringBuilder cleanTibetanTerm = new StringBuilder();
        StringBuilder cleanTibetanDef = new StringBuilder();
        
        // Process English column - remove any non-English content
        String[] englishWords = english.split("\\s+");
        for (String word : englishWords) {
            if (word.matches("[a-zA-Z][a-zA-Z'-]*")) {
                if (cleanEnglish.length() > 0) cleanEnglish.append(" ");
                cleanEnglish.append(word);
            } else if (word.matches("^[a-z]{2,6}$") && cleanSubject.length() == 0) {
                // This might be a misplaced subject code
                cleanSubject.append(word);
            }
        }
        
        // Process subject column - extract only valid subject codes
        String[] subjectWords = subject.split("\\s+");
        for (String word : subjectWords) {
            if (word.matches("^[a-z]{2,6}$") || word.matches("^[a-z]+,[a-z]+$")) {
                if (cleanSubject.length() > 0) cleanSubject.append(" ");
                cleanSubject.append(word);
            } else if (word.matches("[a-zA-Z][a-zA-Z'-]*") && !word.matches(".*[\\u0F00-\\u0FFF].*")) {
                // Misplaced English word
                if (cleanEnglish.length() > 0) cleanEnglish.append(" ");
                cleanEnglish.append(word);
            }
        }
        
        // Process Tibetan content - combine and then properly split
        String allTibetanContent = (tibetanTerm + " " + tibetanDef).trim();
        
        // Remove any English/Latin content that leaked into Tibetan columns
        StringBuilder cleanTibetanContent = new StringBuilder();
        String[] tibetanWords = allTibetanContent.split("\\s+");
        for (String word : tibetanWords) {
            if (word.matches(".*[\\u0F00-\\u0FFF].*") || // Contains Tibetan
                word.matches("[།༏༎༑༔]") || // Tibetan punctuation
                !word.matches(".*[a-zA-Z].*")) { // No Latin letters
                
                if (cleanTibetanContent.length() > 0) cleanTibetanContent.append(" ");
                cleanTibetanContent.append(word);
            } else if (word.matches("^[a-z]{2,6}$") && cleanSubject.length() == 0) {
                // Misplaced subject code
                cleanSubject.append(word);
            } else if (word.matches("[a-zA-Z][a-zA-Z'-]*")) {
                // Misplaced English word
                if (cleanEnglish.length() > 0) cleanEnglish.append(" ");
                cleanEnglish.append(word);
            }
        }
        
        // Now properly split the clean Tibetan content into term and definition
        String[] tibetanParts = reconstructTibetanContent(cleanTibetanContent.toString());
        cleanTibetanTerm.append(tibetanParts[0]);
        cleanTibetanDef.append(tibetanParts[1]);
        
        String[] result = {
            cleanEnglish.toString().trim(),
            cleanSubject.toString().trim(), 
            cleanTibetanTerm.toString().trim(),
            cleanTibetanDef.toString().trim()
        };
        
        System.out.println("[RECONSTRUCT_ALL] Result - English: '" + result[0] + "' Subject: '" + result[1] + 
            "' Term: '" + result[2] + "' Def: '" + result[3].substring(0, Math.min(30, result[3].length())) + "...'");
        
        return result;
    }
    
    private static boolean isValidColumnTerm(ColumnizedTerm term) {
        if (term == null) return false;
        
        String english = term.englishTerm;
        String tibetan = term.tibetanTerm;
        
        if (english == null || tibetan == null) return false;
        if (english.trim().length() < 2 || tibetan.trim().length() < 2) return false;
        if (english.length() > 100) return false;
        
        // Must contain actual Tibetan characters
        if (!tibetan.matches(".*[\\u0F00-\\u0FFF].*")) return false;
        
        // Must be reasonable English
        if (!english.matches("[a-zA-Z][a-zA-Z\\s'-]*")) return false;
        
        return true;
    }
}