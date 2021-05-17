package backprop;


import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.colors.XChartSeriesColors;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.XChartSeriesMarkers;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Main {

    public static void main(String[] args) {

//        File file = new File("wine.data");
//        double[][] dataArray = DataParser.getDataArray(file);
//
////        dataArray = DataParser.getClassInSingleIndex(dataArray, 27, 34);
//        dataArray = DataParser.normalizeArray(dataArray, 1, 14, false);
////        dataArray = DataParser.normalizeOutputSingleClass(dataArray, 0, 1);
//
//        double[][] input = DataParser.getInput2(dataArray, 0, dataArray.length);
//        double[][] target = DataParser.getOutput2(dataArray, 0, dataArray.length);
//
//        double[] testInput = input[0];
//        double[] testTarget = target[0];
//
//        File logFile = new File("log.txt");
//
////        Network network = new Network();
//
//        backpropagationCrossValidate(new int[]{13, 20, 3}, 1, false, input, target, 5_000, 0.001);
//
////        backpropagation(new int[]{13, 20, 3}, 1, false, input, target, 5_000, 0.001, 10, logFile);

        File file = new File("Faults.NNA");
        double[][] dataArray = DataParser.getDataArray(file);

//        dataArray = DataParser.getClassInSingleIndex(dataArray, 27, 34);
        dataArray = DataParser.normalizeArray(dataArray, 0, 26, false);
//        dataArray = DataParser.normalizeOutputSingleClass(dataArray, 0, 1);

        double[][] input = DataParser.getInput(dataArray, 0, dataArray.length);
        double[][] target = DataParser.getOutput(dataArray, 0, dataArray.length);

        File logFile = new File("log.txt");

        long startTime = System.currentTimeMillis();

        ArrayList<Double> xV = new ArrayList<>();
        ArrayList<Double> yV = new ArrayList<>();

        int counter = 0;
        for (double lr = 0.0001; lr <= 1; lr *= 10) {
            xV.add(lr);
            yV.add(backpropagationCrossValidate(new int[]{27, 50, 40, 7}, 1, false, input, target, 500, lr, 10));

            counter++;
        }

        long stopTime = System.currentTimeMillis();

        double[] xVals = new double[xV.size()];
        double[] yVals = new double[yV.size()];

        for (int i = 0; i < xV.size(); i++) {
            xVals[i] = xV.get(i);
            yVals[i] = yV.get(i);
        }



        XYChart xyChart = new XYChartBuilder()
                .width(800)
                .height(600)
                .theme(Styler.ChartTheme.Matlab)
                .title("PK(lr)")
                .xAxisTitle("Learning rate")
                .yAxisTitle("PK")
                .build();

        XYSeries series1 = xyChart.addSeries("Target", xVals, yVals);

        JFrame chartFrame = new SwingWrapper(xyChart).displayChart();


        double seconds = (stopTime - startTime) / 1000.0;
        double minutes = (double) ((int) seconds / 60);

        System.out.println("Complete Time:" + minutes + " minutes, " + (seconds - 60 * minutes) + " seconds.");

//        backpropagationCrossValidate(new int[]{27, 50, 40, 7}, 1, false, input, target, 500, 0.01, 5);

//        backpropagation(new int[]{27, 50, 40, 7}, 1, false, input, target, 4000, 0.005, 6, logFile);
    }

    public static double validateNetwork(Network network, double[][] input, double[][] target) {
        int valid = 0;
        for (int j = 0; j < input.length; j++) {
            double[] outputVal = network.calculateOutput(input[j]);

            if (DataSet.getSingleClass(outputVal) == DataSet.getSingleClass(target[j])) {
                valid++;
            }

//                    double error = calculateSquaredError(outputVal, dataParts.get(testingSet).get(j).getTarget());
//                    sumOfSquaredErrorsTest += error;
        }

        double validPercent = ((double) valid / input.length * 100);

        System.out.println("valid%:" + validPercent);

        return validPercent;
    }

    public static double backpropagationCrossValidate(int[] networkLayers, double beta, boolean bipolarActivation, double[][] input, double[][] target,
                                                      int maxEpoch, double learningRate, int k) {

        ArrayList<DataSet> data = new ArrayList<>();

        for (int i = 0; i < input.length; i++) {
            data.add(new DataSet(input[i], target[i]));
        }

        Collections.shuffle(data);

        //number of parts of the data
//        int k = 10;

        //sizes of each data part
        int singleSize = data.size() / k;
        int lastSize = singleSize + data.size() - singleSize * k;

//        System.out.println(singleSize + " " + lastSize);

        ArrayList<ArrayList<DataSet>> dataParts = new ArrayList<>();

        //creating the data parts
        int counter = 0;
        for (int i = 0; i < k; i++) {
            dataParts.add(new ArrayList<>());

            if (i == k - 1) {
                for (int j = 0; j < lastSize; j++) {
                    dataParts.get(i).add(data.get(counter));
                    counter++;
                }
            } else {
                for (int j = 0; j < singleSize; j++) {
                    dataParts.get(i).add(data.get(counter));
                    counter++;
                }
            }
        }

        double sumOfValid = 0;

        long startTime = System.currentTimeMillis();

        for (int testingSet = 0; testingSet < k; testingSet++) {

            Network network = new Network(networkLayers, beta, bipolarActivation);

            for (int trainingSet = 0; trainingSet < k; trainingSet++) {

                if (trainingSet == testingSet)
                    continue;

                //training on a single training set
                for (int epoch = 0; epoch < maxEpoch; epoch++) {

                    double sumOfSquaredErrors = 0;
                    for (int j = 0; j < dataParts.get(trainingSet).size(); j++) {
                        network.train(dataParts.get(trainingSet).get(j).getInput(), dataParts.get(trainingSet).get(j).getTarget(), learningRate);

//                        double[] outputVal = network.output[network.NETWORK_SIZE - 1];
//                        double error = calculateSquaredError(outputVal, dataParts.get(trainingSet).get(j).getTarget());
//                        sumOfSquaredErrors += error;

                    }

//                    System.out.println("Training set:" + trainingSet + " MSE:" + (sumOfSquaredErrors / dataParts.get(trainingSet).size()) );
                }
            }

            //testing on the testing set
//                double sumOfSquaredErrorsTest = 0;
            int valid = 0;
            for (int j = 0; j < dataParts.get(testingSet).size(); j++) {
                double[] outputVal = network.calculateOutput(dataParts.get(testingSet).get(j).getInput());

                if (DataSet.getSingleClass(outputVal) == DataSet.getSingleClass(dataParts.get(testingSet).get(j).getTarget())) {
                    valid++;
                }

//                    double error = calculateSquaredError(outputVal, dataParts.get(testingSet).get(j).getTarget());
//                    sumOfSquaredErrorsTest += error;
            }

            double validPercent = ((double) valid / dataParts.get(testingSet).size() * 100);

            System.out.println("Testing set:" + testingSet + " valid%:" + validPercent);

            sumOfValid += validPercent;

        }

        long stopTime = System.currentTimeMillis();

        double seconds = (stopTime - startTime) / 1000.0;
        double minutes = (double) ((int) seconds / 60);

        System.out.println("Average valid percent:" + sumOfValid / k);
        System.out.println("Time:" + minutes + " minutes, " + (seconds - 60 * minutes) + " seconds.");

        return sumOfValid / k;

    }

    public static double backpropagation(int[] networkLayers, double beta, boolean bipolarActivation,
                                         double[][] input, double[][] target, int maxEpoch, double learningRate, int maxValidations, File logFile) {

        Network network = new Network(networkLayers, beta, bipolarActivation);

        // Splitting data
        ArrayList<DataSet> data = new ArrayList<>();
        for (int i = 0; i < input.length; i++) {
            data.add(new DataSet(input[i], target[i]));
        }

        Collections.shuffle(data);

        ArrayList<DataSet> trainData = new ArrayList<>();
        ArrayList<DataSet> testData = new ArrayList<>();
        ArrayList<DataSet> validationData = new ArrayList<>();

        for (int i = 0; i < (int) (0.8 * data.size()); i++) {
            trainData.add(data.get(i));
        }

        for (int i = (int) (0.8 * data.size()); i < (int) (0.9 * data.size()); i++) {
            testData.add(data.get(i));
        }

        for (int i = (int) (0.9 * data.size()); i < data.size(); i++) {
            validationData.add(data.get(i));
        }

//        Collections.sort(trainData);
        Collections.sort(testData);

        int printCounter = 0;

        // Creating arrays with MSE chart values
        ArrayList<Double> xMSE = new ArrayList<>();
        ArrayList<Double> yMSETraining = new ArrayList<>();
        ArrayList<Double> yMSEValidating = new ArrayList<>();

        // Creating arrays with chart values
        double[] xVals = new double[testData.size()];
        double[][] yVals = new double[2][];

        yVals[0] = new double[testData.size()];
        yVals[1] = new double[testData.size()];

        for (int i = 0; i < testData.size(); i++) {
            xVals[i] = i;
            yVals[0][i] = DataSet.getSingleClass(testData.get(i).getTarget());
        }

        // Creating output chart
        XYChart xyChart = new XYChartBuilder()
                .width(800)
                .height(600)
                .theme(Styler.ChartTheme.Matlab)
                .title("Output of the testing data set")
                .xAxisTitle("Test record")
                .yAxisTitle("Class")
                .build();

        XYSeries series1 = xyChart.addSeries("Target", xVals, yVals[0]);
        XYSeries series2 = xyChart.addSeries("Output", xVals, yVals[1]);

        series1.setLineWidth(2);
        series1.setLineColor(XChartSeriesColors.GREEN);
        series1.setMarkerColor(XChartSeriesColors.GREEN);

        series2.setLineWidth(1);
        series2.setLineColor(XChartSeriesColors.RED);
        series2.setMarker(XChartSeriesMarkers.CIRCLE);
        series2.setMarkerColor(XChartSeriesColors.RED);
        series2.setLineStyle(SeriesLines.DASH_DASH);

        JFrame chartFrame = new SwingWrapper(xyChart).displayChart();

        // Training
        long startTime = System.currentTimeMillis();

        double lastValidationMSE = 0;
//        int maxValidations = 6;
        int currentValidations = 0;

        int epoch = 0;
        for (epoch = 0; epoch < maxEpoch; epoch++) {
            double sumOfSquaredErrors = 0;
            for (int j = 0; j < trainData.size(); j++) {
                network.train(trainData.get(j).getInput(), trainData.get(j).getTarget(), learningRate);

                double[] outputVal = network.output[network.NETWORK_SIZE - 1];
                double error = calculateSquaredError(outputVal, trainData.get(j).getTarget());
                sumOfSquaredErrors += error;

            }
            printCounter++;

            if (printCounter > 10) {
                double sumOfSquaredErrorsValidation = 0;

                // Update test data chart
                for (int j = 0; j < testData.size(); j++) {
                    double[] outputVal = network.calculateOutput(testData.get(j).getInput());

                    yVals[1][j] = DataSet.getSingleClass(outputVal);
                }
                xyChart.updateXYSeries("Output", xVals, yVals[1], null);

                // Check % valid in test data
                int valid = 0;
                for (int i = 0; i < testData.size(); i++) {
                    if (yVals[0][i] == yVals[1][i]) {
                        valid++;
                    }
                }

                chartFrame.revalidate();
                chartFrame.repaint();

                // Add new point to the MSE chart
                double MSE = sumOfSquaredErrors / trainData.size();
                xMSE.add((double) epoch);
                yMSETraining.add(MSE);


                // Calculate SSE and MSE for validation data
                for (int j = 0; j < validationData.size(); j++) {
                    double[] outputVal = network.calculateOutput(validationData.get(j).getInput());
                    double error = calculateSquaredError(outputVal, validationData.get(j).getTarget());
                    sumOfSquaredErrorsValidation += error;
                }

                double MSEValidation = sumOfSquaredErrorsValidation / validationData.size();

                yMSEValidating.add(MSEValidation);

                if (MSEValidation > lastValidationMSE) {
                    currentValidations++;
                } else {
                    currentValidations = 0;
                }

                if (currentValidations > maxValidations) {
                    System.out.println("Max validations limit exceeded! Breaking!");
                    break;
                }

                lastValidationMSE = MSEValidation;

                System.out.println("Epoch:" + epoch + " SSE:" + sumOfSquaredErrors + " MSE:" + MSE + " SSEvalidation:" + sumOfSquaredErrorsValidation
                        + " MSEvalidation:" + MSEValidation + " % valid:" + (double) valid / testData.size() * 100);
                printCounter = 0;
            }
        }

        long stopTime = System.currentTimeMillis();

        // Creating arrays for MSE chart
        double[] xMSEVals = new double[xMSE.size()];
        double[] yMSEValsTraining = new double[yMSETraining.size()];
        double[] yMSEValsValidating = new double[yMSEValidating.size()];

        for (int i = 0; i < xMSE.size(); i++) {
            xMSEVals[i] = xMSE.get(i);
        }
        for (int i = 0; i < yMSETraining.size(); i++) {
            yMSEValsTraining[i] = yMSETraining.get(i);
        }
        for (int i = 0; i < yMSEValidating.size(); i++) {
            yMSEValsValidating[i] = yMSEValidating.get(i);
        }

        XYChart xyChartMSE = new XYChartBuilder().width(800).height(600).theme(Styler.ChartTheme.Matlab).title("MSE").xAxisTitle("Epoch").yAxisTitle("MSE").build();

        XYSeries series1MSETraining = xyChartMSE.addSeries("MSE Training", xMSEVals, yMSEValsTraining);
        XYSeries series2MSEValidating = xyChartMSE.addSeries("MSE Validating", xMSEVals, yMSEValsValidating);

//        series1MSETraining.setLineWidth(2);
//        series1MSETraining.setLineColor(XChartSeriesColors.GREEN);
//        series1MSETraining.setMarkerColor(XChartSeriesColors.GREEN);
//
//        series1MSETraining.setLineWidth(1);
//        series1MSETraining.setLineColor(XChartSeriesColors.RED);
//        series1MSETraining.setMarker(XChartSeriesMarkers.CIRCLE);
//        series1MSETraining.setMarkerColor(XChartSeriesColors.RED);
//        series1MSETraining.setLineStyle(SeriesLines.DASH_DASH);

        JFrame chartFrameMSE = new SwingWrapper(xyChartMSE).displayChart();


        double sumOfSquaredErrors = 0;

        int valid = 0;

        // Outputting logs to file
        try (PrintWriter pw = new PrintWriter(logFile)) {
            for (int j = 0; j < testData.size(); j++) {
                double[] outputVal = network.calculateOutput(testData.get(j).getInput());
                double error = calculateSquaredError(outputVal, testData.get(j).getTarget());
                sumOfSquaredErrors += error;
                pw.println("------");
                pw.println("Input data:" + j);
                pw.println("Input:" + printArray(testData.get(j).getInput()));
                pw.println("output:" + printArray(outputVal));
                pw.println("target:" + printArray(testData.get(j).getTarget()));
                pw.println("error:" + String.format("%.5f", error));
                pw.println("------");

                xVals[j] = j;
                yVals[1][j] = DataSet.getSingleClass(outputVal);
            }

            for (int i = 0; i < testData.size(); i++) {
                if (yVals[0][i] == yVals[1][i]) {
                    valid++;
                }
            }
            double seconds = (stopTime - startTime) / 1000.0;
            double minutes = (double) ((int) seconds / 60);

            pw.print("For the TEST DATA SET:");
            pw.println("Network size:" + Arrays.toString(network.NETWORK_LAYERS_SIZES));
            pw.println("Max epoch:" + maxEpoch);
            pw.println("Final epoch:" + epoch);
            pw.println("Learning rate:" + learningRate);
            pw.println("Time:" + minutes + " minutes, " + (seconds - 60 * minutes) + "seconds.");
            pw.println("Test data SSE:" + sumOfSquaredErrors);
            pw.println("Test data MSE:" + sumOfSquaredErrors / testData.size());
            pw.println("Test data % valid:" + (double) valid / testData.size() * 100);
        } catch (IOException e) {
            e.printStackTrace();
            return 0;
        }

        return (double) valid / testData.size();

    }

    public static String printArray(double[] array) {
        StringBuilder sB = new StringBuilder();

        sB.append("[");
        for (int i = 0; i < array.length; i++) {
            sB.append(String.format("%.5f", array[i]));
            if (i < array.length - 1) {
                sB.append(", ");
            }
        }
        sB.append("]");

        return sB.toString();
    }

    public static double calculateSquaredError(double[] output, double[] target) {
        double error = 0;

        for (int i = 0; i < output.length; i++) {
            error += (target[i] - output[i]) * (target[i] - output[i]);
        }

        error /= output.length;

        return error;
    }
}
