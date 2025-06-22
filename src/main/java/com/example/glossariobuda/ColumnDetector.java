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
        
        // Split text into lines for spatial analysis
        String[] lines = ocrText.split("\\n");
        
        for (String line : lines) {
            if (line.trim().isEmpty()) continue;
            
            ColumnizedTerm term = analyzeLineColumns(line);
            if (term != null) {
                terms.add(term);
                System.out.println("[COLUMN] " + term);
            }
        }
        
        return terms;
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
        // Split by 3+ spaces (likely column boundaries)
        String[] parts = line.split("\\s{3,}");
        
        if (parts.length >= 3) {
            System.out.println("[SPACING] Found " + parts.length + " parts");
            for (int i = 0; i < parts.length; i++) {
                System.out.println("  Part " + i + ": '" + parts[i] + "'");
            }
            
            String english = parts[0].trim();
            String subject = "";
            String tibetanTerm = "";
            String tibetanDef = "";
            
            // Analyze the parts
            if (parts.length >= 4) {
                // 4 clear columns
                subject = parts[1].trim();
                tibetanTerm = parts[2].trim();
                tibetanDef = parts[3].trim();
            } else if (parts.length == 3) {
                // Need to split the middle part or last part
                if (isSubjectCode(parts[1])) {
                    subject = parts[1].trim();
                    // parts[2] contains both Tibetan term and definition
                    String[] tibetanParts = splitTibetanColumns(parts[2]);
                    tibetanTerm = tibetanParts[0];
                    tibetanDef = tibetanParts[1];
                } else {
                    // parts[1] might be mixed English + subject
                    String[] englishSubject = splitEnglishAndSubject(parts[1]);
                    if (!englishSubject[1].isEmpty()) {
                        english = (english + " " + englishSubject[0]).trim();
                        subject = englishSubject[1];
                        String[] tibetanParts = splitTibetanColumns(parts[2]);
                        tibetanTerm = tibetanParts[0];
                        tibetanDef = tibetanParts[1];
                    }
                }
            }
            
            if (isValidTerm(english) && containsTibetan(tibetanTerm)) {
                return new ColumnizedTerm(english, subject, tibetanTerm, tibetanDef);
            }
        }
        
        return null;
    }
    
    private static ColumnizedTerm detectByPattern(String line) {
        // Pattern: English words + subject code + Tibetan text
        Pattern pattern = Pattern.compile(
            "^([a-zA-Z][a-zA-Z\\s-']+?)\\s+" +           // English term
            "([a-z]{2,6})\\s+" +                         // Subject code
            "([\\u0F00-\\u0FFF].*)$"                     // All Tibetan content
        );
        
        Matcher matcher = pattern.matcher(line);
        if (matcher.find()) {
            String english = matcher.group(1).trim();
            String subject = matcher.group(2).trim();
            String allTibetan = matcher.group(3).trim();
            
            String[] tibetanParts = splitTibetanColumns(allTibetan);
            String tibetanTerm = tibetanParts[0];
            String tibetanDef = tibetanParts[1];
            
            System.out.println("[PATTERN] English: '" + english + "' Subject: '" + subject + "' Tibetan: '" + allTibetan.substring(0, Math.min(30, allTibetan.length())) + "...'");
            
            return new ColumnizedTerm(english, subject, tibetanTerm, tibetanDef);
        }
        
        return null;
    }
    
    private static ColumnizedTerm detectByCharacterTransition(String line) {
        // Find the transition from English to Tibetan characters
        int tibetanStart = -1;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c >= '\u0F00' && c <= '\u0FFF') {
                tibetanStart = i;
                break;
            }
        }
        
        if (tibetanStart > 0) {
            String englishPart = line.substring(0, tibetanStart).trim();
            String tibetanPart = line.substring(tibetanStart).trim();
            
            // Extract subject from English part
            String[] englishSubject = splitEnglishAndSubject(englishPart);
            String english = englishSubject[0];
            String subject = englishSubject[1];
            
            // Split Tibetan part
            String[] tibetanParts = splitTibetanColumns(tibetanPart);
            String tibetanTerm = tibetanParts[0];
            String tibetanDef = tibetanParts[1];
            
            if (isValidTerm(english)) {
                System.out.println("[TRANSITION] English: '" + english + "' Subject: '" + subject + "'");
                return new ColumnizedTerm(english, subject, tibetanTerm, tibetanDef);
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
    
    private static String[] splitTibetanColumns(String tibetanText) {
        String cleaned = tibetanText.trim();
        
        // Method 1: Look for རྐྱེན། which often ends terms
        int rkyenIndex = cleaned.indexOf("རྐྱེན།");
        if (rkyenIndex > 0) {
            String beforeRkyen = cleaned.substring(0, rkyenIndex);
            String afterRkyen = cleaned.substring(rkyenIndex + 5).trim();
            
            // Find the actual term boundary
            String[] words = beforeRkyen.trim().split("\\s+");
            if (words.length >= 2) {
                String lastWord = words[words.length - 1];
                if (lastWord.endsWith("པའི་") || lastWord.endsWith("པའེ་")) {
                    String term = lastWord + "རྐྱེན།";
                    String remainingStart = String.join(" ", java.util.Arrays.copyOfRange(words, 0, words.length - 1));
                    String definition = (remainingStart + " " + afterRkyen).trim();
                    return new String[]{term, definition};
                }
            }
        }
        
        // Method 2: Split on first shad
        int firstShad = cleaned.indexOf('།');
        if (firstShad > 0 && firstShad < cleaned.length() - 1) {
            String term = cleaned.substring(0, firstShad + 1).trim();
            String definition = cleaned.substring(firstShad + 1).trim();
            
            if (term.length() >= 3 && term.length() <= 30 && definition.length() > 5) {
                return new String[]{term, definition};
            }
        }
        
        // Method 3: Split on double spaces
        String[] spaceParts = cleaned.split("\\s{2,}");
        if (spaceParts.length >= 2) {
            return new String[]{spaceParts[0].trim(), String.join(" ", java.util.Arrays.copyOfRange(spaceParts, 1, spaceParts.length))};
        }
        
        // Fallback: take first few words as term
        String[] words = cleaned.split("\\s+");
        if (words.length > 3) {
            String term = String.join(" ", java.util.Arrays.copyOfRange(words, 0, Math.min(3, words.length)));
            String definition = String.join(" ", java.util.Arrays.copyOfRange(words, 3, words.length));
            return new String[]{term, definition};
        }
        
        return new String[]{cleaned, ""};
    }
    
    private static boolean isSubjectCode(String text) {
        return text != null && text.matches("^[a-z]{2,6}$");
    }
    
    private static boolean isValidTerm(String text) {
        return text != null && text.length() >= 3 && text.matches(".*[a-zA-Z].*");
    }
    
    private static boolean containsTibetan(String text) {
        return text != null && text.matches(".*[\\u0F00-\\u0FFF].*");
    }
}