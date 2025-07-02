package com.example.glossariobuda;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.rendering.ImageType;

public class PDFToTIFF {
    public static void main(String[] args) throws Exception {
        String pdfPath = "C:/Users/tashi.TASHI-LENOVO/APPS/GlossarioBUDA/3-4.pdf";
        PDDocument document = Loader.loadPDF(new File(pdfPath));
        PDFRenderer renderer = new PDFRenderer(document);

        for (int i = 0; i < document.getNumberOfPages(); ++i) {
            BufferedImage image = renderer.renderImageWithDPI(i, 300, ImageType.RGB);
            File output = new File("output_page_" + i + ".tif");
            ImageIO.write(image, "TIFF", output);  // May require TIFF plugin
        }

        document.close();
        System.out.println("Done.");
    }
}
