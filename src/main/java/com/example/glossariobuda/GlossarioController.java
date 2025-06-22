package com.example.glossariobuda;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

import java.io.File;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class GlossarioController implements Initializable {
    
    @FXML private TextField searchField;
    @FXML private ListView<DatabaseManager.Term> resultsListView;
    @FXML private TextArea detailsArea;
    @FXML private Label statusLabel;
    @FXML private Button clearButton;
    @FXML private Button addTermButton;
    @FXML private Button editButton;
    @FXML private Button deleteButton;
    @FXML private Button exportButton;
    @FXML private Button ocrButton;
    @FXML private Button recentButton;
    
    private DatabaseManager dbManager;
    private ExportManager exportManager;
    private OCRProcessor ocrProcessor;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        initializeManagers();
        setupEventHandlers();
    }
    
    private void initializeManagers() {
        dbManager = new DatabaseManager();
        exportManager = new ExportManager(dbManager);
        ocrProcessor = new OCRProcessor(dbManager);
    }
    
    private void setupEventHandlers() {
        searchField.setOnKeyReleased(e -> performSearch());
        resultsListView.getSelectionModel().selectedItemProperty().addListener(
            (obs, oldVal, newVal) -> displayTermDetails(newVal)
        );
    }
    
    @FXML
    private void clearSearch() {
        searchField.clear();
        resultsListView.getItems().clear();
        detailsArea.clear();
        statusLabel.setText("Pronto. Digite um termo para buscar.");
    }
    
    @FXML
    private void showAddTermDialog() {
        Dialog<DatabaseManager.Term> dialog = new Dialog<>();
        dialog.setTitle("Adicionar Novo Termo");
        dialog.setHeaderText("Adicione um novo termo ao glossário");
        
        ButtonType addButtonType = new ButtonType("Adicionar", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(addButtonType, ButtonType.CANCEL);
        
        GridPane grid = createTermEditGrid(null);
        dialog.getDialogPane().setContent(grid);
        
        // Get form fields
        TextField sourceTermField = (TextField) grid.getChildren().get(1);
        ComboBox<String> sourceLangCombo = (ComboBox<String>) grid.getChildren().get(3);
        TextField targetTermField = (TextField) grid.getChildren().get(5);
        ComboBox<String> targetLangCombo = (ComboBox<String>) grid.getChildren().get(7);
        TextArea contextArea = (TextArea) grid.getChildren().get(9);
        TextField contributorField = (TextField) grid.getChildren().get(11);
        
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == addButtonType) {
                dbManager.addTerm(
                    sourceTermField.getText(),
                    sourceLangCombo.getValue(),
                    targetTermField.getText(),
                    targetLangCombo.getValue(),
                    contextArea.getText(),
                    contributorField.getText(),
                    null
                );
                statusLabel.setText("Termo adicionado com sucesso!");
                performSearch(); // Refresh results
                return null;
            }
            return null;
        });
        
        dialog.showAndWait();
    }
    
    @FXML
    private void editSelectedTerm() {
        DatabaseManager.Term selectedTerm = resultsListView.getSelectionModel().getSelectedItem();
        if (selectedTerm == null) {
            showAlert("Aviso", "Selecione um termo para editar.");
            return;
        }
        
        Dialog<DatabaseManager.Term> dialog = new Dialog<>();
        dialog.setTitle("Editar Termo");
        dialog.setHeaderText("Edite o termo selecionado");
        
        ButtonType saveButtonType = new ButtonType("Salvar", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(saveButtonType, ButtonType.CANCEL);
        
        GridPane grid = createTermEditGrid(selectedTerm);
        dialog.getDialogPane().setContent(grid);
        
        // Get form fields
        TextField sourceTermField = (TextField) grid.getChildren().get(1);
        ComboBox<String> sourceLangCombo = (ComboBox<String>) grid.getChildren().get(3);
        TextField targetTermField = (TextField) grid.getChildren().get(5);
        ComboBox<String> targetLangCombo = (ComboBox<String>) grid.getChildren().get(7);
        TextArea contextArea = (TextArea) grid.getChildren().get(9);
        TextField contributorField = (TextField) grid.getChildren().get(11);
        ComboBox<String> statusCombo = (ComboBox<String>) grid.getChildren().get(13);
        
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == saveButtonType) {
                dbManager.updateTerm(
                    selectedTerm.getId(),
                    sourceTermField.getText(),
                    sourceLangCombo.getValue(),
                    targetTermField.getText(),
                    targetLangCombo.getValue(),
                    contextArea.getText(),
                    contributorField.getText(),
                    null,
                    statusCombo.getValue()
                );
                statusLabel.setText("Termo atualizado com sucesso!");
                performSearch(); // Refresh results
                return null;
            }
            return null;
        });
        
        dialog.showAndWait();
    }
    
    @FXML
    private void deleteSelectedTerm() {
        DatabaseManager.Term selectedTerm = resultsListView.getSelectionModel().getSelectedItem();
        if (selectedTerm == null) {
            showAlert("Aviso", "Selecione um termo para deletar.");
            return;
        }
        
        Alert confirmAlert = new Alert(Alert.AlertType.CONFIRMATION);
        confirmAlert.setTitle("Confirmar Exclusão");
        confirmAlert.setHeaderText("Deletar termo?");
        confirmAlert.setContentText("Tem certeza que deseja deletar o termo '" + selectedTerm.getSourceTerm() + "'?");
        
        confirmAlert.showAndWait().ifPresent(response -> {
            if (response == ButtonType.OK) {
                dbManager.deleteTerm(selectedTerm.getId());
                statusLabel.setText("Termo deletado com sucesso!");
                performSearch(); // Refresh results
            }
        });
    }
    
    @FXML
    private void showExportDialog() {
        Dialog<String> dialog = new Dialog<>();
        dialog.setTitle("Exportar para WhatsApp");
        dialog.setHeaderText("Escolha o tipo de exportação");
        
        ButtonType exportButtonType = new ButtonType("Exportar", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(exportButtonType, ButtonType.CANCEL);
        
        VBox content = new VBox(10);
        content.setPadding(new Insets(10));
        
        RadioButton recentRadio = new RadioButton("Termos dos últimos 7 dias");
        RadioButton allRadio = new RadioButton("Todos os termos");
        recentRadio.setSelected(true);
        
        ToggleGroup group = new ToggleGroup();
        recentRadio.setToggleGroup(group);
        allRadio.setToggleGroup(group);
        
        content.getChildren().addAll(
            new Label("Selecione o que exportar:"),
            recentRadio,
            allRadio
        );
        
        dialog.getDialogPane().setContent(content);
        
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == exportButtonType) {
                String export = recentRadio.isSelected() 
                    ? exportManager.exportRecentTermsForWhatsApp(7)
                    : exportManager.exportAllTermsForWhatsApp();
                
                showExportResult(export);
            }
            return null;
        });
        
        dialog.showAndWait();
    }
    
    @FXML
    private void showOCRDialog() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Selecionar PDF para OCR");
        fileChooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("PDF Files", "*.pdf")
        );
        
        File selectedFile = fileChooser.showOpenDialog(null);
        if (selectedFile != null) {
            processOCR(selectedFile.getAbsolutePath());
        }
    }
    
    @FXML
    private void showRecentTerms() {
        List<DatabaseManager.Term> recentTerms = dbManager.getRecentTerms(20);
        resultsListView.getItems().clear();
        resultsListView.getItems().addAll(recentTerms);
        statusLabel.setText("Mostrando os 20 termos mais recentes");
        
        if (!recentTerms.isEmpty()) {
            resultsListView.getSelectionModel().selectFirst();
        }
    }
    
    private void performSearch() {
        String searchText = searchField.getText().trim();
        if (searchText.isEmpty()) {
            resultsListView.getItems().clear();
            detailsArea.clear();
            statusLabel.setText("Digite um termo para buscar.");
            return;
        }
        
        List<DatabaseManager.Term> results = dbManager.searchTerms(searchText);
        resultsListView.getItems().clear();
        resultsListView.getItems().addAll(results);
        
        statusLabel.setText("Encontrados " + results.size() + " resultados para '" + searchText + "'");
        
        if (!results.isEmpty()) {
            resultsListView.getSelectionModel().selectFirst();
        } else {
            detailsArea.clear();
        }
    }
    
    private void displayTermDetails(DatabaseManager.Term term) {
        if (term == null) {
            detailsArea.clear();
            return;
        }
        
        StringBuilder details = new StringBuilder();
        details.append("🔤 Termo Original: ").append(term.getSourceTerm()).append("\n");
        details.append("🌍 Idioma Original: ").append(term.getSourceLanguage()).append("\n\n");
        details.append("📝 Tradução: ").append(term.getTargetTerm()).append("\n");
        details.append("🌍 Idioma Tradução: ").append(term.getTargetLanguage()).append("\n\n");
        
        if (term.getContext() != null && !term.getContext().trim().isEmpty()) {
            details.append("📖 Contexto: ").append(term.getContext()).append("\n\n");
        }
        
        if (term.getContributor() != null && !term.getContributor().trim().isEmpty()) {
            details.append("👤 Contribuidor: ").append(term.getContributor()).append("\n");
        }
        
        details.append("📅 Data: ").append(term.getDateAdded()).append("\n");
        details.append("✓ Status: ").append(term.getVerifiedStatus()).append("\n");
        
        if (term.getNotes() != null && !term.getNotes().trim().isEmpty()) {
            details.append("📌 Notas: ").append(term.getNotes()).append("\n");
        }
        
        detailsArea.setText(details.toString());
    }
    
    private GridPane createTermEditGrid(DatabaseManager.Term term) {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20, 150, 10, 10));
        
        TextField sourceTermField = new TextField();
        sourceTermField.setPromptText("śamatha");
        ComboBox<String> sourceLangCombo = new ComboBox<>();
        sourceLangCombo.getItems().addAll("Sanskrit", "Tibetan", "Pali", "English", "Portuguese");
        sourceLangCombo.setValue("English");
        
        TextField targetTermField = new TextField();
        targetTermField.setPromptText("meditação posicionada");
        ComboBox<String> targetLangCombo = new ComboBox<>();
        targetLangCombo.getItems().addAll("Tibetan", "Portuguese", "English", "Spanish");
        targetLangCombo.setValue("Tibetan");
        
        TextArea contextArea = new TextArea();
        contextArea.setPromptText("Contexto ou fonte da tradução...");
        contextArea.setPrefRowCount(3);
        
        TextField contributorField = new TextField();
        contributorField.setPromptText("Seu nome");
        
        ComboBox<String> statusCombo = new ComboBox<>();
        statusCombo.getItems().addAll("draft", "reviewed", "verified", "approved");
        statusCombo.setValue("draft");
        
        // Populate fields if editing existing term
        if (term != null) {
            sourceTermField.setText(term.getSourceTerm());
            sourceLangCombo.setValue(term.getSourceLanguage());
            targetTermField.setText(term.getTargetTerm());
            targetLangCombo.setValue(term.getTargetLanguage());
            contextArea.setText(term.getContext());
            contributorField.setText(term.getContributor());
            statusCombo.setValue(term.getVerifiedStatus());
        }
        
        grid.add(new Label("Termo Original:"), 0, 0);
        grid.add(sourceTermField, 1, 0);
        grid.add(new Label("Idioma Original:"), 0, 1);
        grid.add(sourceLangCombo, 1, 1);
        grid.add(new Label("Tradução:"), 0, 2);
        grid.add(targetTermField, 1, 2);
        grid.add(new Label("Idioma Tradução:"), 0, 3);
        grid.add(targetLangCombo, 1, 3);
        grid.add(new Label("Contexto:"), 0, 4);
        grid.add(contextArea, 1, 4);
        grid.add(new Label("Contribuidor:"), 0, 5);
        grid.add(contributorField, 1, 5);
        grid.add(new Label("Status:"), 0, 6);
        grid.add(statusCombo, 1, 6);
        
        return grid;
    }

    private void processOCR(String pdfPath) {
        statusLabel.setText("Processando texto/OCR... Isso pode demorar alguns minutos.");

        Thread extractionThread = new Thread(() -> {
            // Try the hybrid approach first
            HybridTextExtractor hybridExtractor = new HybridTextExtractor(dbManager);

            hybridExtractor.processPDF(pdfPath, new HybridTextExtractor.ProgressCallback() {
                @Override
                public void onProgress(int current, int total, String message) {
                    Platform.runLater(() -> statusLabel.setText(message));
                }

                @Override
                public void onError(String error) {
                    Platform.runLater(() -> statusLabel.setText("Erro: " + error));
                }

                @Override
                public void onComplete(String message) {
                    Platform.runLater(() -> {
                        statusLabel.setText(message);
                        showAlert("Processamento Completo", message);
                        showRecentTerms();
                    });
                }
            });
        });

        extractionThread.setDaemon(true);
        extractionThread.start();
    }
    
    private void showExportResult(String exportText) {
        Dialog<Void> dialog = new Dialog<>();
        dialog.setTitle("Exportação Completa");
        dialog.setHeaderText("Copie o texto abaixo para o WhatsApp:");
        
        TextArea textArea = new TextArea(exportText);
        textArea.setEditable(false);
        textArea.setWrapText(true);
        textArea.setPrefRowCount(15);
        textArea.setPrefColumnCount(50);
        
        dialog.getDialogPane().setContent(textArea);
        dialog.getDialogPane().getButtonTypes().add(ButtonType.CLOSE);
        
        dialog.showAndWait();
    }
    
    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
    
    public void cleanup() {
        if (dbManager != null) {
            dbManager.close();
        }
    }
}