package com.naaturel.ANN.infrastructure.graph;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class GraphVisualizer {

    XYSeriesCollection dataset;

    public GraphVisualizer(){
        this.dataset = new XYSeriesCollection();
    }

    public void addPoint(String title, float x, float y) {
        if (this.dataset.getSeriesIndex(title) == -1)
            this.dataset.addSeries(new XYSeries(title));
        this.dataset.getSeries(title).add(x, y);
    }

    public void addEquation(String title, float y1, float y2, float k, float xMin, float xMax) {
        for (float x1 = xMin; x1 <= xMax; x1 += 0.01f) {
            float x2 = (-y1 * x1 - k) / y2;
            addPoint(title, x1, x2);
        }
    }

    public void buildLineGraph(){
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Model learning", "X", "Y", dataset
        );
        JFrame frame = new JFrame("Training Loss");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }


    public void buildScatterGraph(int lower, int upper){
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Predictions", "X", "Y", dataset
        );
        XYPlot plot = chart.getXYPlot();
        plot.getDomainAxis().setRange(lower, upper);
        plot.getRangeAxis().setRange(lower, upper);

        JFrame frame = new JFrame("Predictions");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

}
