package com.example.glossariobuda;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 4-Column OCR Parser for Buddhist Glossary
 * 
 * Handles the specific 4-column structure:
 * Column 1: English Term (can span 2 lines)
 * Column 2: Subject Code  
 * Column 3: Tibetan Term
 * Column 4: Tibetan Definition (can span multiple lines)
 */
public class OCRProcessor_4column {
    
    public static class TermEntry {
        public final String englishTerm;
        public final String tibetanTerm;
        public final String tibetanDefinition;
        public final String subject;
        
        public TermEntry(String englishTerm, String tibetanTerm, String tibetanDefinition, String subject) {
            this.englishTerm = englishTerm;
            this.tibetanTerm = tibetanTerm;
            this.tibetanDefinition = tibetanDefinition;
            this.subject = subject;
        }
    }
    
    public static List<TermEntry> extractFourColumnTerms(String ocrText) {
        List<TermEntry> terms = new ArrayList<>();
        
        // Pattern 1: Abbreviations format (acc. Accountancy ཚད་རིགས།)
        Pattern abbreviationPattern = Pattern.compile(
            "^([a-zA-Z]+\\.)\\s+" +                      // Abbreviation (e.g., "acc.", "chem.")
            "([A-Za-z][A-Za-z\\s-']+?)\\s+" +           // Full English term
            "([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།ག-ཟ\\s]+)", // Tibetan (may include definition)
            Pattern.MULTILINE
        );
        
        // Pattern 2: Main glossary 4-column structure
        Pattern fourColumnPattern = Pattern.compile(
            // English term (may be split: "absent\nmindedness")
            "([a-zA-Z][a-zA-Z\\s-']*?)(?:\\s*\\n([a-zA-Z][a-zA-Z\\s-']*?))?\\s+" +
            // Subject code (bot, psycho, med, etc.)
            "([a-z]{2,6})\\s+" +
            // All Tibetan text (OCR concatenated columns 3+4)
            "([\\u0F00-\\u0FFF][\\u0F00-\\u0FFF\\s་།ག-ཟ\\.\\s]+)",
            Pattern.MULTILINE | Pattern.DOTALL
        );

        // Try abbreviations pattern first
        Matcher matcher = abbreviationPattern.matcher(ocrText);
        while (matcher.find()) {
            String abbreviation = matcher.group(1);
            String englishTerm = cleanTerm(matcher.group(2));
            String tibetanText = matcher.group(3);
            
            // For abbreviations, there's usually no separate definition
            String[] tibetanParts = separateTibetanColumns(tibetanText);
            String tibetanTerm = tibetanParts[0];
            String tibetanDefinition = tibetanParts[1];

            if (isValidEntry(englishTerm, tibetanTerm)) {
                terms.add(new TermEntry(englishTerm, tibetanTerm, tibetanDefinition, "abbrev"));
                System.out.println("[ABBREV] " + abbreviation + " = " + englishTerm + " -> " + tibetanTerm);
            }
        }
        
        // Try main glossary 4-column pattern
        matcher = fourColumnPattern.matcher(ocrText);
        while (matcher.find()) {
            // Reconstruct English term
            String englishPart1 = cleanTerm(matcher.group(1));
            String englishPart2 = matcher.group(2) != null ? cleanTerm(matcher.group(2)) : "";
            String englishTerm = (englishPart1 + " " + englishPart2).trim();
            
            String subject = matcher.group(3);
            String concatenatedTibetan = matcher.group(4);
            
            // Separate Tibetan term from definition
            String[] tibetanParts = separateTibetanColumns(concatenatedTibetan);
            String tibetanTerm = tibetanParts[0];
            String tibetanDefinition = tibetanParts[1];

            if (isValidEntry(englishTerm, tibetanTerm)) {
                terms.add(new TermEntry(englishTerm, tibetanTerm, tibetanDefinition, subject));
                System.out.println("[4-COL] " + englishTerm + " [" + subject + "] -> " + tibetanTerm);
                if (!tibetanDefinition.isEmpty()) {
                    System.out.println("     Def: " + tibetanDefinition.substring(0, Math.min(50, tibetanDefinition.length())) + "...");
                }
            }
        }
        
        return terms;
    }
    
    private static String[] separateTibetanColumns(String concatenatedTibetan) {
        String cleaned = concatenatedTibetan.trim();
        System.out.println("[DEBUG] Separating: " + cleaned);
        
        // Method 1: Look for རྐྱེན། pattern which often ends terms
        // Handle case: "སྐྱེ་ལྡན་མིན་པའི་ other_text རྐྱེན།"
        int rkyenIndex = cleaned.indexOf("རྐྱེན།");
        if (rkyenIndex > 0) {
            // Find the word boundary before རྐྱེན།
            String beforeRkyen = cleaned.substring(0, rkyenIndex);
            String afterRkyen = cleaned.substring(rkyenIndex + 5); // རྐྱེན། is 5 chars
            
            // Look for the actual term that should end with རྐྱེན།
            String[] beforeWords = beforeRkyen.trim().split("\\s+");
            if (beforeWords.length >= 2) {
                // Take last few words + རྐྱེན། as the term
                String lastWord = beforeWords[beforeWords.length - 1];
                String secondLastWord = beforeWords.length > 1 ? beforeWords[beforeWords.length - 2] : "";
                
                // Reconstruct term: should be something like "སྐྱེ་ལྡན་མིན་པའི་རྐྱེན།"
                String reconstructedTerm;
                if (secondLastWord.endsWith("པའི་") || secondLastWord.endsWith("པའེ་")) {
                    reconstructedTerm = secondLastWord + "རྐྱེན།";
                    // Definition is everything before secondLastWord + everything after
                    String definitionStart = String.join(" ", java.util.Arrays.copyOfRange(beforeWords, 0, beforeWords.length - 2));
                    String definition = (definitionStart + " " + lastWord + " " + afterRkyen).trim();
                    return new String[]{reconstructedTerm, definition};
                } else {
                    reconstructedTerm = lastWord + "རྐྱེན།";
                    String definitionStart = String.join(" ", java.util.Arrays.copyOfRange(beforeWords, 0, beforeWords.length - 1));
                    String definition = (definitionStart + " " + afterRkyen).trim();
                    return new String[]{reconstructedTerm, definition};
                }
            }
        }
        
        // Method 2: Look for first shad (།) to separate term from definition
        int firstShad = cleaned.indexOf('།');
        if (firstShad > 0 && firstShad < cleaned.length() - 1) {
            String term = cleaned.substring(0, firstShad + 1).trim();
            String definition = cleaned.substring(firstShad + 1).trim();
            
            // Validate: term should be reasonable length
            if (term.length() >= 5 && term.length() <= 30 && definition.length() > 5) {
                System.out.println("[DEBUG] Shad split - Term: " + term + " | Def: " + definition.substring(0, Math.min(30, definition.length())) + "...");
                return new String[]{term, definition};
            }
        }
        
        // Method 3: Split on multiple spaces (column separation)
        String[] spaceSplit = cleaned.split("\\s{3,}");
        if (spaceSplit.length >= 2) {
            System.out.println("[DEBUG] Space split - Term: " + spaceSplit[0] + " | Def: " + spaceSplit[1].substring(0, Math.min(30, spaceSplit[1].length())) + "...");
            return new String[]{spaceSplit[0].trim(), String.join(" ", java.util.Arrays.copyOfRange(spaceSplit, 1, spaceSplit.length))};
        }
        
        // Method 4: Take first 2-3 Tibetan syllables as term, rest as definition
        String[] syllables = cleaned.split("\\s+");
        if (syllables.length > 4) {
            // Look for natural break point (common term endings)
            for (int i = 2; i < Math.min(5, syllables.length); i++) {
                String syllable = syllables[i];
                if (syllable.endsWith("།") || syllable.endsWith("པའི") || syllable.endsWith("པའེ") || 
                    syllable.endsWith("བའི") || syllable.endsWith("རྐྱེན")) {
                    String term = String.join(" ", java.util.Arrays.copyOfRange(syllables, 0, i + 1));
                    String definition = String.join(" ", java.util.Arrays.copyOfRange(syllables, i + 1, syllables.length));
                    System.out.println("[DEBUG] Syllable split - Term: " + term + " | Def: " + definition.substring(0, Math.min(30, definition.length())) + "...");
                    return new String[]{term, definition};
                }
            }
            
            // Fallback: first 3 syllables as term
            String term = String.join(" ", java.util.Arrays.copyOfRange(syllables, 0, 3));
            String definition = String.join(" ", java.util.Arrays.copyOfRange(syllables, 3, syllables.length));
            System.out.println("[DEBUG] Fallback split - Term: " + term + " | Def: " + definition.substring(0, Math.min(30, definition.length())) + "...");
            return new String[]{term, definition};
        }
        
        // Last resort: entire text as term
        System.out.println("[DEBUG] No split - using entire text as term");
        return new String[]{cleaned, ""};
    }
    
    private static String cleanTerm(String term) {
        if (term == null) return "";
        return term.trim()
                .replaceAll("\\s+", " ")
                .replaceAll("^[^a-zA-Z\\u0F00-\\u0FFF]+|[^a-zA-Z\\u0F00-\\u0FFF\\s'་།-]+$", "")
                .toLowerCase();
    }
    
    private static boolean isValidEntry(String englishTerm, String tibetanTerm) {
        if (englishTerm == null || tibetanTerm == null) return false;
        if (englishTerm.length() < 3 || englishTerm.length() > 50) return false;
        if (tibetanTerm.length() < 2) return false;
        
        // Must contain actual Tibetan characters
        if (!tibetanTerm.matches(".*[\\u0F00-\\u0FFF].*")) return false;
        
        // Must be valid English
        if (!englishTerm.matches("[a-zA-Z][a-zA-Z\\s'-]*")) return false;
        
        // Skip document headers
        String[] skipWords = {"abbreviations", "glossary", "terms", "page", "tibetan", "serial"};
        for (String skip : skipWords) {
            if (englishTerm.equals(skip)) return false;
        }
        
        return true;
    }
}