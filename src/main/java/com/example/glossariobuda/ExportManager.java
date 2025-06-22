package com.example.glossariobuda;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

public class ExportManager {
    private final DatabaseManager dbManager;
    
    public ExportManager(DatabaseManager dbManager) {
        this.dbManager = dbManager;
    }
    
    public String exportRecentTermsForWhatsApp(int days) {
        LocalDateTime cutoffDate = LocalDateTime.now().minusDays(days);
        String cutoffString = cutoffDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        
        List<DatabaseManager.Term> recentTerms = dbManager.getTermsAddedAfter(cutoffString);
        
        if (recentTerms.isEmpty()) {
            return "📚 *Glossário BUDA* - Nenhum termo novo nos últimos " + days + " dias";
        }
        
        StringBuilder export = new StringBuilder();
        export.append("📚 *Glossário BUDA*\n");
        export.append("🗓️ Novos termos dos últimos ").append(days).append(" dias\n");
        export.append("═══════════════════════\n\n");
        
        String currentLanguagePair = "";
        for (DatabaseManager.Term term : recentTerms) {
            String languagePair = term.getSourceLanguage() + " → " + term.getTargetLanguage();
            
            if (!languagePair.equals(currentLanguagePair)) {
                if (!currentLanguagePair.isEmpty()) {
                    export.append("\n");
                }
                export.append("🌍 *").append(languagePair.toUpperCase()).append("*\n");
                export.append("───────────────\n");
                currentLanguagePair = languagePair;
            }
            
            export.append("• **").append(term.getSourceTerm()).append("**\n");
            export.append("  ➜ ").append(term.getTargetTerm()).append("\n");
            
            if (term.getContext() != null && !term.getContext().trim().isEmpty()) {
                export.append("  📝 _").append(term.getContext()).append("_\n");
            }
            
            if (term.getContributor() != null && !term.getContributor().trim().isEmpty()) {
                export.append("  👤 ").append(term.getContributor()).append("\n");
            }
            
            export.append("\n");
        }
        
        export.append("═══════════════════════\n");
        export.append("📊 Total: ").append(recentTerms.size()).append(" novos termos\n");
        export.append("🕐 Gerado em: ").append(LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm")));
        
        return export.toString();
    }
    
    public String exportAllTermsForWhatsApp() {
        List<DatabaseManager.Term> allTerms = dbManager.getRecentTerms(1000); // Get up to 1000 terms
        
        if (allTerms.isEmpty()) {
            return "📚 *Glossário BUDA* - Nenhum termo no banco de dados";
        }
        
        StringBuilder export = new StringBuilder();
        export.append("📚 *GLOSSÁRIO COMPLETO BUDA*\n");
        export.append("═══════════════════════════\n\n");
        
        String currentLanguagePair = "";
        for (DatabaseManager.Term term : allTerms) {
            String languagePair = term.getSourceLanguage() + " → " + term.getTargetLanguage();
            
            if (!languagePair.equals(currentLanguagePair)) {
                if (!currentLanguagePair.isEmpty()) {
                    export.append("\n");
                }
                export.append("🌍 *").append(languagePair.toUpperCase()).append("*\n");
                export.append("───────────────────\n");
                currentLanguagePair = languagePair;
            }
            
            export.append("• **").append(term.getSourceTerm()).append("** → ").append(term.getTargetTerm());
            
            String status = getStatusIcon(term.getVerifiedStatus());
            export.append(" ").append(status).append("\n");
            
            if (term.getContext() != null && !term.getContext().trim().isEmpty()) {
                export.append("  📝 _").append(term.getContext()).append("_\n");
            }
        }
        
        export.append("\n═══════════════════════════\n");
        export.append("📊 Total: ").append(allTerms.size()).append(" termos\n");
        export.append("🕐 Gerado em: ").append(LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm")));
        
        return export.toString();
    }
    
    public boolean saveExportToFile(String content, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write(content);
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
    
    public String exportTermsByContributor(String contributor) {
        List<DatabaseManager.Term> allTerms = dbManager.getRecentTerms(1000);
        List<DatabaseManager.Term> contributorTerms = allTerms.stream()
            .filter(term -> contributor.equals(term.getContributor()))
            .toList();
        
        if (contributorTerms.isEmpty()) {
            return "📚 *Glossário BUDA* - Nenhum termo encontrado para " + contributor;
        }
        
        StringBuilder export = new StringBuilder();
        export.append("📚 *Glossário BUDA*\n");
        export.append("👤 Contribuições de: ").append(contributor).append("\n");
        export.append("═══════════════════════\n\n");
        
        for (DatabaseManager.Term term : contributorTerms) {
            export.append("• **").append(term.getSourceTerm()).append("**\n");
            export.append("  ➜ ").append(term.getTargetTerm()).append("\n");
            
            if (term.getContext() != null && !term.getContext().trim().isEmpty()) {
                export.append("  📝 _").append(term.getContext()).append("_\n");
            }
            
            export.append("  📅 ").append(term.getDateAdded()).append("\n\n");
        }
        
        export.append("═══════════════════════\n");
        export.append("📊 Total: ").append(contributorTerms.size()).append(" termos\n");
        export.append("🕐 Gerado em: ").append(LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm")));
        
        return export.toString();
    }
    
    private String getStatusIcon(String status) {
        return switch (status != null ? status.toLowerCase() : "draft") {
            case "verified", "approved" -> "✅";
            case "reviewed" -> "🔍";
            case "draft" -> "📝";
            default -> "❓";
        };
    }
    
    public String generateWhatsAppMessage(String exportContent) {
        return exportContent + "\n\n" +
               "🔗 _Compartilhe este glossário com outros tradutores!_\n" +
               "💡 _Para sugerir correções ou novos termos, entre em contato._";
    }
}