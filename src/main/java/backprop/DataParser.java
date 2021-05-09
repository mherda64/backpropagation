package backprop;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class DataParser {

    public static void main(String[] args) {
        File file = new File("Faults.NNA");
        double[][] dataArray = getDataArray(file);
        dataArray = normalizeArray(dataArray, 0, 26, false);
        dataArray = getClassInSingleIndex(dataArray, 27, 34);

        double[][] input = getInput(dataArray, 0, 1941);
        double[][] output = getOutputSingleClass(dataArray, 0, 1941);


        File inputFile = new File("input.csv");
        saveToCSV(input, inputFile);

        File outputFile = new File("output.csv");
        saveToCSV(output, outputFile);
    }

    public static double normalize(double x, double xMin, double xMax) {
        //taken newMax = 1 and newMin = -1
        return ((x - xMin) * 2) / (xMax - xMin) - 1;
    }

    public static double normalize(double x, double xMin, double xMax, double newMin, double newMax) {
        //taken newMax = 1 and newMin = -1
        return ((newMax - newMin) * (x - xMin) * 2) / (xMax - xMin) + newMin;
    }

    public static double[][] getClassInSingleIndex(double[][] data, int firstIndex, int lastIndex) {
        int numberOfClasses = firstIndex == 0 ? lastIndex - firstIndex + 1 : lastIndex - firstIndex;
        double[][] output = new double[data.length][data[0].length - numberOfClasses + 1];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < firstIndex; j++) {
                output[i][j] = data[i][j];
            }

            for (int j = firstIndex; j < data[i].length; j++) {
                if(data[i][j] == 1) {
                    output[i][firstIndex] = (double) (j - firstIndex);
                    break;
                }
            }
        }

        return output;
    }

    public static boolean saveToCSV(double[][] data, File file) {
        Writer out;
        try (PrintWriter pw = new PrintWriter(file)) {
            for (double[] singleLine : data) {
                pw.println(convertToCSV(singleLine));
            }
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }

    private static String convertToCSV(double[] array) {
        StringBuilder sB = new StringBuilder();

        for (int i = 0; i < array.length; i++) {
            sB.append(array[i]);
            if (i < array.length - 1) {
                sB.append(",");
            }
        }

        return sB.toString();
    }

    public static double[][] normalizeArray(double[][] data, int indexFrom, int indexTo, boolean singleClass) {
        double[][] output = new double[data.length][data[0].length];
        for (int i = 0; i < data[0].length; i++) {
            if (i >= indexFrom && i <= indexTo) {
//                System.out.println("column:" + i + " Max:" + getMax(data, i) + " Min:" + getMin(data, i));
                double max = getMax(data, i);
                double min = getMin(data, i);

                for (int j = 0; j < data.length; j++) {
                    output[j][i] = normalize(data[j][i], min, max);
                }
            } else {
                if (singleClass) {
                    double max = 6.0;
                    double min = 0.0;

                    for (int j = 0; j < data.length; j++) {
                        output[j][i] = normalize(data[j][i], min, max, 0, 0.5);
                    }
                } else {
                    for (int j = 0; j < data.length; j++) {
                        output[j][i] = data[j][i];
                    }
                }
            }

        }

        return output;
    }

    public static double[][] normalizeOutputSingleClass(double[][] data, double newMin, double newMax) {
        double[][] output = new double[data.length][data[0].length];
//                System.out.println("column:" + i + " Max:" + getMax(data, i) + " Min:" + getMin(data, i));
        double max = getMax(data, 27);
        double min = getMin(data, 27);

        for (int j = 0; j < data.length; j++) {
            output[j][27] = normalize(data[j][27], min, max, newMin, newMax);
        }


        return output;
    }


    private static double getMin(double[][] data, int column) {
        double min = data[0][column];

        for (int i = 0; i < data.length; i++) {
            if (data[i][column] < min) {
                min = data[i][column];
            }
        }

        return min;
    }

    private static double getMax(double[][] data, int column) {
        double max = data[0][column];

        for (int i = 0; i < data.length; i++) {
            if (data[i][column] > max) {
                max = data[i][column];
            }
        }

        return max;
    }

    public static double[][] getDataArray(File file) {
        ArrayList<double[]> values = new ArrayList<>();
        int arrayCounter = 0;
        //for each line from the data file
        try (Scanner scanner = new Scanner(file)) {
            while (scanner.hasNextLine()) {
                //parse every line to a Double[] array
                //and add it to the values ArrayList
                String[] lineValues = scanner.nextLine().split("\t|,");
                values.add(new double[lineValues.length]);
                int innerCounter = 0;
                for (String single : lineValues) {
                    values.get(arrayCounter)[innerCounter] = Double.parseDouble(single);
                    innerCounter++;
                }
                arrayCounter++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        //move the ArrayList to Double[][] array
        double[][] outputArray = new double[arrayCounter][];
        values.toArray(outputArray);
        return outputArray;
    }

    public static double[][] getOutputSingleClass(double[][] data, int beginning, int end) {
        double[][] output = new double[end - beginning][1];

        int counter = 0;
        for (int i = beginning; i < end; i++) {
            output[counter][0] = data[i][data[i].length - 1];
            counter++;
        }

        return output;
    }

    public static double[][] getInputSingleClass(double[][] data, int beginning, int end) {
        double[][] output = new double[end - beginning][data[0].length - 1];

        int counter = 0;
        for (int i = beginning; i < end; i++) {
            for (int j = 0; j < data[0].length - 1; j++) {
                output[counter][j] = data[i][j];
            }
            counter++;
        }

        return output;
    }

    public static double[][] getOutput(double[][] data, int beginning, int end) {
        double[][] output = new double[end - beginning][7];

        int counter = 0;
        for (int i = beginning; i < end; i++) {
            for (int j = 0; j < 7; j++) {
                output[counter][j] = data[i][data[i].length - 7 + j];
            }
            counter++;
        }

        return output;
    }

    public static double[][] getInput(double[][] data, int beginning, int end) {
        double[][] output = new double[end - beginning][data[0].length - 7];

        int counter = 0;
        for (int i = beginning; i < end; i++) {
            for (int j = 0; j < data[0].length - 7; j++) {
                output[counter][j] = data[i][j];
            }
            counter++;
        }

        return output;
    }

    public static double[][] getOutput2(double[][] data, int beginning, int end) {
        double[][] output = new double[end - beginning][3];

        int counter = 0;
        for (int i = beginning; i < end; i++) {
            if (data[i][0] == 1.0) {
                output[counter][0] = 1;
                output[counter][1] = 0;
                output[counter][2] = 0;
            } else if (data[i][0] == 2.0) {
                output[counter][0] = 0;
                output[counter][1] = 1;
                output[counter][2] = 0;
            } else if (data[i][0] == 3.0) {
                output[counter][0] = 0;
                output[counter][1] = 0;
                output[counter][2] = 1;
            }
//            output[counter][0] = data[i][0];
            counter++;
        }

        return output;
    }

    public static double[][] getInput2(double[][] data, int beginning, int end) {
        double[][] output = new double[end - beginning][data[0].length - 1];

        int counter = 0;
        for (int i = beginning; i < end; i++) {
            for (int j = 0; j < data[0].length - 1; j++) {
                output[counter][j] = data[i][j + 1];
            }
            counter++;
        }

        return output;
    }

}
