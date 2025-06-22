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

    private static void debugPrintSpaces(String line) {
        StringBuilder sb = new StringBuilder();
        int spaceRun = 0;

        for (char c : line.toCharArray()) {
            if (c == ' ') {
                spaceRun++;
            } else {
                if (spaceRun > 0) {
                    sb.append("␣".repeat(spaceRun));  // use ␣ to visualize spaces
                    spaceRun = 0;
                }
                sb.append(c);
            }
        }
        if (spaceRun > 0) {
            sb.append("␣".repeat(spaceRun));
        }

        System.out.println("[SPACE VISUAL] " + sb.toString());
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
        System.out.println("[PROCESS] Processing: " + line);
        debugPrintSpaces(line); // Optional: prints positions of space groups

        TermBuilder newTerm = parseNewTermLine(line);
        if (newTerm.hasEnglish()) {
            System.out.println("  -> NEW TERM detected");
            terms.add(newTerm);
            System.out.println("  -> Added: " + newTerm);
        } else if (!terms.isEmpty()) {
            System.out.println("  -> CONTINUATION line");
            appendToLastTerm(line, terms.get(terms.size() - 1));
        } else {
            System.out.println("  -> LINE IGNORED: No valid English term or previous entry");
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
        
        System.out.println("  -> Analyzing line for column boundaries: " + line);
        
        // Method 1: Try to find patterns based on character transitions
        // Look for English -> Subject code -> Tibetan transitions
        
        // Find first English word end
        int firstWordEnd = -1;
        for (int i = 0; i < line.length(); i++) {
            if (line.charAt(i) == ' ' && firstWordEnd == -1) {
                // Look ahead to see if next part looks like another English word or subject code
                String nextPart = "";
                int j = i + 1;
                while (j < line.length() && line.charAt(j) != ' ') {
                    nextPart += line.charAt(j);
                    j++;
                }
                
                // If next part looks like a subject code (2-6 lowercase letters)
                if (nextPart.matches("^[a-z]{2,6}$") || nextPart.matches("^[a-z]+,[a-z]+$")) {
                    firstWordEnd = i;
                    boundaries.add(i);
                    System.out.println("    -> Found English/Subject boundary at: " + i + " (next: '" + nextPart + "')");
                    
                    // Find subject code end
                    int subjectEnd = j;
                    if (subjectEnd < line.length()) {
                        boundaries.add(subjectEnd);
                        System.out.println("    -> Found Subject/Tibetan boundary at: " + subjectEnd);
                    }
                    break;
                }
            }
        }
        
        // Method 2: If no clear subject code found, look for English -> Tibetan transition
        if (boundaries.isEmpty()) {
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                if (c >= '\u0F00' && c <= '\u0FFF') {
                    // Found first Tibetan character, look backwards for space
                    for (int j = i - 1; j >= 0; j--) {
                        if (line.charAt(j) == ' ') {
                            boundaries.add(j);
                            System.out.println("    -> Found English/Tibetan boundary at: " + j);
                            break;
                        }
                    }
                    break;
                }
            }
        }
        
        // Method 3: Fallback - look for any spaces that might be column separators
        if (boundaries.isEmpty()) {
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
                    if (inSpace && spaceCount >= 1) { // Reduced from 3 to 1
                        boundaries.add(spaceStart);
                        System.out.println("    -> Found space boundary at: " + spaceStart + " (count: " + spaceCount + ")");
                    }
                    inSpace = false;
                    spaceCount = 0;
                }
            }
        }
        
        System.out.println("  -> Final boundaries: " + boundaries);
        return boundaries;
    }
    
    private static String[] splitTibetanContent(String tibetanText) {
        String cleaned = tibetanText.trim();
        System.out.println("  -> Splitting Tibetan content: " + cleaned.substring(0, Math.min(50, cleaned.length())) + "...");
        
        // Method 1: Look for རྐྱེན། pattern - this is the key completion marker
        int rkyenIndex = cleaned.indexOf("རྐྱེན།");
        if (rkyenIndex >= 0) {
            System.out.println("  -> Found རྐྱེན། at position: " + rkyenIndex);
            
            // Find the term that should end with རྐྱེན།
            // Look for pattern: "word ending with པའི་/པའེ་/བའི་/བའེ་" + "རྐྱེན།"
            String beforeRkyen = cleaned.substring(0, rkyenIndex);
            String afterRkyen = cleaned.substring(rkyenIndex + 5).trim(); // 5 = length of "རྐྱེན།"
            
            System.out.println("  -> Before རྐྱེན།: '" + beforeRkyen + "'");
            System.out.println("  -> After རྐྱེན།: '" + afterRkyen.substring(0, Math.min(30, afterRkyen.length())) + "...'");
            
            // Find the word that should be completed with རྐྱེན།
            String[] beforeWords = beforeRkyen.trim().split("\\s+");
            for (int i = beforeWords.length - 1; i >= 0; i--) {
                String word = beforeWords[i];
                if (word.endsWith("པའི་") || word.endsWith("པའེ་") || 
                    word.endsWith("བའི་") || word.endsWith("བའེ་") ||
                    word.endsWith("མིན་པའེ་") || word.endsWith("མིན་པའི་")) {
                    
                    // This word + རྐྱེན། forms the complete term
                    String completeTerm = word + "རྐྱེན།";
                    
                    // Everything else is definition
                    String definitionPart = "";
                    if (i > 0) {
                        definitionPart = String.join(" ", java.util.Arrays.copyOfRange(beforeWords, 0, i)) + " ";
                    }
                    definitionPart += afterRkyen;
                    
                    System.out.println("  -> Completed term: '" + completeTerm + "'");
                    System.out.println("  -> Definition: '" + definitionPart.substring(0, Math.min(30, definitionPart.length())) + "...'");
                    
                    return new String[]{completeTerm, definitionPart.trim()};
                }
            }
            
            // Fallback: if no clear completion pattern, take last word + རྐྱེན།
            if (beforeWords.length > 0) {
                String lastWord = beforeWords[beforeWords.length - 1];
                String completeTerm = lastWord + "རྐྱེན།";
                String definitionPart = "";
                if (beforeWords.length > 1) {
                    definitionPart = String.join(" ", java.util.Arrays.copyOfRange(beforeWords, 0, beforeWords.length - 1)) + " ";
                }
                definitionPart += afterRkyen;
                
                System.out.println("  -> Fallback term: '" + completeTerm + "'");
                return new String[]{completeTerm, definitionPart.trim()};
            }
        }
        
        // Method 2: Split on first shad (།) - common punctuation marker
        int firstShad = cleaned.indexOf('།');
        if (firstShad > 0 && firstShad < cleaned.length() - 1) {
            String term = cleaned.substring(0, firstShad + 1).trim();
            String definition = cleaned.substring(firstShad + 1).trim();
            
            // Validate term length is reasonable
            if (term.length() >= 3 && term.length() <= 50 && definition.length() > 5) {
                System.out.println("  -> Split on shad: '" + term + "' | '" + definition.substring(0, Math.min(30, definition.length())) + "...'");
                return new String[]{term, definition};
            }
        }
        
        // Method 3: Take first few words as term (fallback)
        String[] words = cleaned.split("\\s+");
        if (words.length > 3) {
            String term = String.join(" ", java.util.Arrays.copyOfRange(words, 0, Math.min(3, words.length)));
            String definition = String.join(" ", java.util.Arrays.copyOfRange(words, 3, words.length));
            System.out.println("  -> Fallback split: '" + term + "' | '" + definition.substring(0, Math.min(30, definition.length())) + "...'");
            return new String[]{term, definition};
        }
        
        // No split possible
        System.out.println("  -> No split possible, returning as single term");
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
