package com.example.glossariobuda;

import java.util.ArrayList;
import java.util.List;

/**
 * Line-by-Line Column Parser for Buddhist Glossary
 * 
 * Strategy:
 * 1. Find English marker -> put into data structure
 * 2. Find extra space -> move to subject portion -> put into data structure  
 * 3. Find extra space -> move to translation portion -> put into data structure
 * 4. Find space -> move to definition portion -> put into data structure
 * 5. Find new line -> look for English, match with existing entry to concatenate
 * 6. Continue for every line, concatenating to respective portions
 * 7. At the end, put everything together
 */
public class LineByLineColumnParser {
    
    public static class TermBuilder {
        public String englishTerm = "";
        public String subject = "";
        public String tibetanTerm = "";
        public String tibetanDefinition = "";
        public boolean expectingCompletion = false; // Waiting for རྐྱེན། or other completion
        
        public boolean hasEnglish() {
            return !englishTerm.trim().isEmpty();
        }
        
        public boolean isComplete() {
            return hasEnglish() && !tibetanTerm.trim().isEmpty() && !expectingCompletion;
        }
        
        public boolean isIncomplete() {
            return hasEnglish() && expectingCompletion;
        }
        
        @Override
        public String toString() {
            return String.format("'%s' [%s] -> '%s' | Def: '%s' %s", 
                englishTerm.trim(), subject.trim(), tibetanTerm.trim(), 
                tibetanDefinition.length() > 40 ? tibetanDefinition.substring(0, 40) + "..." : tibetanDefinition,
                expectingCompletion ? "(INCOMPLETE)" : "");
        }
    }
    
    public static List<TermBuilder> parseColumns(String ocrText) {
        System.out.println("[PARSER] Starting line-by-line column parsing");
        
        List<TermBuilder> terms = new ArrayList<>();
        String[] lines = ocrText.split("\\n");
        
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (line.isEmpty()) continue;
            
            System.out.println("[LINE " + i + "] Processing: " + line);
            
            // Parse this line and either create new term or append to existing
            processLine(line, terms);
        }
        
        System.out.println("[PARSER] Found " + terms.size() + " terms");
        return terms;
    }
    
    private static void processLine(String line, List<TermBuilder> terms) {
        // Step 1: Check if line starts with English (new term)
        if (startsWithEnglish(line)) {
            System.out.println("  -> NEW TERM detected");
            TermBuilder newTerm = parseNewTermLine(line);
            if (newTerm.hasEnglish()) {
                terms.add(newTerm);
                System.out.println("  -> Added: " + newTerm);
            }
        } else {
            // Step 2: This is a continuation line, append to last term
            if (!terms.isEmpty()) {
                System.out.println("  -> CONTINUATION line");
                appendToLastTerm(line, terms.get(terms.size() - 1));
            }
        }
    }
    
    private static TermBuilder parseNewTermLine(String line) {
        TermBuilder term = new TermBuilder();
        
        // Find column positions by detecting large spaces
        List<Integer> spacePositions = findColumnBoundaries(line);
        
        System.out.println("  -> Space positions: " + spacePositions);
        
        if (spacePositions.size() >= 1) {
            // Extract English (column 1)
            term.englishTerm = line.substring(0, spacePositions.get(0)).trim();
            System.out.println("  -> English: '" + term.englishTerm + "'");
            
            if (spacePositions.size() >= 2) {
                // Extract Subject (column 2)  
                term.subject = line.substring(spacePositions.get(0), spacePositions.get(1)).trim();
                System.out.println("  -> Subject: '" + term.subject + "'");
                
                if (spacePositions.size() >= 3) {
                    // Extract Tibetan Term (column 3)
                    term.tibetanTerm = line.substring(spacePositions.get(1), spacePositions.get(2)).trim();
                    System.out.println("  -> Tibetan Term: '" + term.tibetanTerm + "'");
                    
                    // Extract Tibetan Definition (column 4 - rest of line)
                    term.tibetanDefinition = line.substring(spacePositions.get(2)).trim();
                    System.out.println("  -> Tibetan Def: '" + term.tibetanDefinition.substring(0, Math.min(30, term.tibetanDefinition.length())) + "...'");
                } else {
                    // Only 2 columns found, everything after subject is Tibetan
                    String allTibetan = line.substring(spacePositions.get(1)).trim();
                    // Try to split Tibetan term from definition
                    String[] tibetanParts = splitTibetanContent(allTibetan);
                    term.tibetanTerm = tibetanParts[0];
                    term.tibetanDefinition = tibetanParts[1];
                    System.out.println("  -> Combined Tibetan split into: '" + term.tibetanTerm + "' | '" + 
                        term.tibetanDefinition.substring(0, Math.min(30, term.tibetanDefinition.length())) + "...'");
                }
            }
        }
        
        return term;
    }
    
    private static void appendToLastTerm(String line, TermBuilder lastTerm) {
        String cleanLine = line.trim();
        
        // CRITICAL: Check if this line contains completion markers like རྐྱེན།
        if (lastTerm.expectingCompletion && containsCompletionMarker(cleanLine)) {
            // Complete the Tibetan term that was waiting
            String completionPart = extractCompletionPart(cleanLine);
            lastTerm.tibetanTerm += completionPart;
            lastTerm.expectingCompletion = false;
            System.out.println("  -> COMPLETED Tibetan term: '" + lastTerm.tibetanTerm + "'");
            
            // Rest of the line goes to definition
            String remainingText = cleanLine.replace(completionPart, "").trim();
            if (!remainingText.isEmpty()) {
                lastTerm.tibetanDefinition += " " + remainingText;
                System.out.println("  -> Added to definition: '" + remainingText.substring(0, Math.min(30, remainingText.length())) + "...'");
            }
            return;
        }
        
        // Check what type of content this continuation line has
        if (containsTibetan(cleanLine)) {
            // This is likely continuing Tibetan definition
            lastTerm.tibetanDefinition += " " + cleanLine;
            System.out.println("  -> Appended to Tibetan def: '" + cleanLine.substring(0, Math.min(30, cleanLine.length())) + "...'");
        } else if (startsWithLowerCase(cleanLine)) {
            // This might be continuing English term
            lastTerm.englishTerm += " " + cleanLine;
            System.out.println("  -> Appended to English: '" + cleanLine + "'");
        } else {
            // Default: append to definition
            lastTerm.tibetanDefinition += " " + cleanLine;
            System.out.println("  -> Appended to definition: '" + cleanLine.substring(0, Math.min(30, cleanLine.length())) + "...'");
        }
    }
    
    private static List<Integer> findColumnBoundaries(String line) {
        List<Integer> boundaries = new ArrayList<>();
        
        // Look for sequences of 3+ spaces (column separators)
        boolean inSpace = false;
        int spaceStart = -1;
        int spaceCount = 0;
        
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            
            if (c == ' ') {
                if (!inSpace) {
                    inSpace = true;
                    spaceStart = i;
                    spaceCount = 1;
                } else {
                    spaceCount++;
                }
            } else {
                if (inSpace && spaceCount >= 3) {
                    // Found a column boundary
                    boundaries.add(spaceStart);
                }
                inSpace = false;
                spaceCount = 0;
            }
        }
        
        return boundaries;
    }
    
    private static String[] splitTibetanContent(String tibetanText) {
        String cleaned = tibetanText.trim();
        
        // Method 1: Look for རྐྱེན། pattern
        int rkyenIndex = cleaned.indexOf("རྐྱེན།");
        if (rkyenIndex > 0) {
            String beforeRkyen = cleaned.substring(0, rkyenIndex);
            String afterRkyen = cleaned.substring(rkyenIndex + 5).trim();
            
            String[] words = beforeRkyen.trim().split("\\s+");
            if (words.length >= 1) {
                String lastWord = words[words.length - 1];
                if (lastWord.endsWith("པའི་") || lastWord.endsWith("པའེ་")) {
                    String term = lastWord + "རྐྱེན།";
                    String definition = "";
                    if (words.length > 1) {
                        definition = String.join(" ", java.util.Arrays.copyOfRange(words, 0, words.length - 1)) + " ";
                    }
                    definition += afterRkyen;
                    return new String[]{term, definition.trim()};
                }
            }
        }
        
        // Method 2: Split on first shad
        int firstShad = cleaned.indexOf('།');
        if (firstShad > 0 && firstShad < cleaned.length() - 1) {
            String term = cleaned.substring(0, firstShad + 1).trim();
            String definition = cleaned.substring(firstShad + 1).trim();
            
            if (term.length() >= 3 && term.length() <= 30) {
                return new String[]{term, definition};
            }
        }
        
        // Method 3: Take first few words as term
        String[] words = cleaned.split("\\s+");
        if (words.length > 3) {
            String term = String.join(" ", java.util.Arrays.copyOfRange(words, 0, Math.min(3, words.length)));
            String definition = String.join(" ", java.util.Arrays.copyOfRange(words, 3, words.length));
            return new String[]{term, definition};
        }
        
        return new String[]{cleaned, ""};
    }
    
    private static boolean startsWithEnglish(String line) {
        if (line.isEmpty()) return false;
        char first = line.charAt(0);
        return (first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z');
    }
    
    private static boolean startsWithLowerCase(String line) {
        if (line.isEmpty()) return false;
        char first = line.charAt(0);
        return (first >= 'a' && first <= 'z');
    }
    

    private static boolean containsTibetan(String text) {
        for (char c : text.toCharArray()) {
            if (c >= '\u0F00' && c <= '\u0FFF') {
                return true;
            }
        }
        return false;
    }

    private static boolean containsCompletionMarker(String text) {
        // Check for common Tibetan completion patterns
        return text.contains("རྐྱེན།") ||
                text.contains("་རྐྱེན།") ||
                text.matches(".*[\\u0F00-\\u0FFF]+།.*"); // Any Tibetan ending with །
    }

    private static String extractCompletionPart(String text) {
        String cleanText = text.trim();

        // Method 1: Extract རྐྱེན། completion
        if (cleanText.contains("རྐྱེན།")) {
            int rkyenIndex = cleanText.indexOf("རྐྱེན།");
            // Look backwards for the start of the completion
            int startIndex = rkyenIndex;
            while (startIndex > 0 && cleanText.charAt(startIndex - 1) != ' ') {
                startIndex--;
            }
            return cleanText.substring(startIndex, rkyenIndex + 5); // Include རྐྱེན།
        }

        // Method 2: Extract first Tibetan word ending with །
        String[] words = cleanText.split("\\s+");
        for (String word : words) {
            if (word.matches(".*[\\u0F00-\\u0FFF]+།")) {
                System.out.println("  -> Found completion marker: '" + word + "'");
                return word;
            }
        }

        // Method 3: Take first Tibetan syllable as completion
        for (String word : words) {
            if (containsTibetan(word)) {
                System.out.println("  -> Using first Tibetan word as completion: '" + word + "'");
                return word;
            }
        }

        return "";
    }
}
