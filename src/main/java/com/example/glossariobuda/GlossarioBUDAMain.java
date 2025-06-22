package com.example.glossariobuda;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class GlossarioBUDAMain extends Application {
    private GlossarioController controller;

    @Override
    public void start(Stage primaryStage) throws Exception {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("glossario-main.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 800, 600); // Laptop-friendly size
        
        controller = fxmlLoader.getController();
        
        primaryStage.setTitle("Glossário BUDA - Sistema de Terminologia Budista");
        primaryStage.setScene(scene);
        primaryStage.setOnCloseRequest(e -> controller.cleanup());
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}