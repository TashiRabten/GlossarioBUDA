package com.example.glossariobuda;

import java.sql.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class DatabaseManager {
    private static final String DB_URL = "jdbc:sqlite:src/main/resources/terms.db";
    private Connection connection;
    
    public DatabaseManager() {
        initializeDatabase();
    }
    
    private void initializeDatabase() {
        try {
            connection = DriverManager.getConnection(DB_URL);
            createTables();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    private void createTables() throws SQLException {
        String createTermsTable = """
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_term TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_term TEXT NOT NULL,
                target_language TEXT NOT NULL,
                context TEXT,
                contributor TEXT,
                date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
                verified_status TEXT DEFAULT 'draft',
                notes TEXT
            )
        """;
        
        try (Statement stmt = connection.createStatement()) {
            stmt.execute(createTermsTable);
        }
    }
    
    public void addTerm(String sourceTerm, String sourceLanguage, String targetTerm, 
                       String targetLanguage, String context, String contributor, String notes) {
        String sql = """
            INSERT INTO terms (source_term, source_language, target_term, target_language, 
                             context, contributor, notes) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """;
        
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            pstmt.setString(1, sourceTerm);
            pstmt.setString(2, sourceLanguage);
            pstmt.setString(3, targetTerm);
            pstmt.setString(4, targetLanguage);
            pstmt.setString(5, context);
            pstmt.setString(6, contributor);
            pstmt.setString(7, notes);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    public List<Term> searchTerms(String searchText) {
        List<Term> results = new ArrayList<>();
        String sql = """
            SELECT * FROM terms 
            WHERE LOWER(source_term) LIKE LOWER(?) 
               OR LOWER(target_term) LIKE LOWER(?)
            ORDER BY date_added DESC
        """;
        
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            String searchPattern = "%" + searchText + "%";
            pstmt.setString(1, searchPattern);
            pstmt.setString(2, searchPattern);
            
            ResultSet rs = pstmt.executeQuery();
            while (rs.next()) {
                Term term = new Term(
                    rs.getInt("id"),
                    rs.getString("source_term"),
                    rs.getString("source_language"),
                    rs.getString("target_term"),
                    rs.getString("target_language"),
                    rs.getString("context"),
                    rs.getString("contributor"),
                    rs.getString("date_added"),
                    rs.getString("verified_status"),
                    rs.getString("notes")
                );
                results.add(term);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        
        return results;
    }
    
    public List<Term> getRecentTerms(int limit) {
        List<Term> results = new ArrayList<>();
        String sql = "SELECT * FROM terms ORDER BY date_added DESC LIMIT ?";
        
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            pstmt.setInt(1, limit);
            
            ResultSet rs = pstmt.executeQuery();
            while (rs.next()) {
                Term term = new Term(
                    rs.getInt("id"),
                    rs.getString("source_term"),
                    rs.getString("source_language"),
                    rs.getString("target_term"),
                    rs.getString("target_language"),
                    rs.getString("context"),
                    rs.getString("contributor"),
                    rs.getString("date_added"),
                    rs.getString("verified_status"),
                    rs.getString("notes")
                );
                results.add(term);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        
        return results;
    }
    
    public List<Term> getTermsAddedAfter(String dateTime) {
        List<Term> results = new ArrayList<>();
        String sql = "SELECT * FROM terms WHERE date_added > ? ORDER BY date_added DESC";
        
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            pstmt.setString(1, dateTime);
            
            ResultSet rs = pstmt.executeQuery();
            while (rs.next()) {
                Term term = new Term(
                    rs.getInt("id"),
                    rs.getString("source_term"),
                    rs.getString("source_language"),
                    rs.getString("target_term"),
                    rs.getString("target_language"),
                    rs.getString("context"),
                    rs.getString("contributor"),
                    rs.getString("date_added"),
                    rs.getString("verified_status"),
                    rs.getString("notes")
                );
                results.add(term);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        
        return results;
    }
    
    public void updateTermStatus(int termId, String status) {
        String sql = "UPDATE terms SET verified_status = ? WHERE id = ?";
        
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            pstmt.setString(1, status);
            pstmt.setInt(2, termId);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    public void updateTerm(int termId, String sourceTerm, String sourceLanguage, 
                          String targetTerm, String targetLanguage, String context, 
                          String contributor, String notes, String status) {
        String sql = """
            UPDATE terms SET 
                source_term = ?, source_language = ?, target_term = ?, target_language = ?,
                context = ?, contributor = ?, notes = ?, verified_status = ?
            WHERE id = ?
        """;
        
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            pstmt.setString(1, sourceTerm);
            pstmt.setString(2, sourceLanguage);
            pstmt.setString(3, targetTerm);
            pstmt.setString(4, targetLanguage);
            pstmt.setString(5, context);
            pstmt.setString(6, contributor);
            pstmt.setString(7, notes);
            pstmt.setString(8, status);
            pstmt.setInt(9, termId);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    public void deleteTerm(int termId) {
        String sql = "DELETE FROM terms WHERE id = ?";
        
        try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
            pstmt.setInt(1, termId);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    public void close() {
        try {
            if (connection != null) {
                connection.close();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    public static class Term {
        private final int id;
        private final String sourceTerm;
        private final String sourceLanguage;
        private final String targetTerm;
        private final String targetLanguage;
        private final String context;
        private final String contributor;
        private final String dateAdded;
        private final String verifiedStatus;
        private final String notes;
        
        public Term(int id, String sourceTerm, String sourceLanguage, String targetTerm,
                   String targetLanguage, String context, String contributor, 
                   String dateAdded, String verifiedStatus, String notes) {
            this.id = id;
            this.sourceTerm = sourceTerm;
            this.sourceLanguage = sourceLanguage;
            this.targetTerm = targetTerm;
            this.targetLanguage = targetLanguage;
            this.context = context;
            this.contributor = contributor;
            this.dateAdded = dateAdded;
            this.verifiedStatus = verifiedStatus;
            this.notes = notes;
        }
        
        // Getters
        public int getId() { return id; }
        public String getSourceTerm() { return sourceTerm; }
        public String getSourceLanguage() { return sourceLanguage; }
        public String getTargetTerm() { return targetTerm; }
        public String getTargetLanguage() { return targetLanguage; }
        public String getContext() { return context; }
        public String getContributor() { return contributor; }
        public String getDateAdded() { return dateAdded; }
        public String getVerifiedStatus() { return verifiedStatus; }
        public String getNotes() { return notes; }
        
        @Override
        public String toString() {
            return sourceTerm + " → " + targetTerm;
        }
    }
}