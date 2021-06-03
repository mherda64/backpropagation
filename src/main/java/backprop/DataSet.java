package backprop;

public class DataSet implements Comparable<DataSet> {

    private double[] input;

    private double[] target;

    public DataSet(double[] input, double[] target) {
        this.input = input;
        this.target = target;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getTarget() {
        return target;
    }

    public void setTarget(double[] target) {
        this.target = target;
    }

    @Override
    public int compareTo(DataSet o) {
        return Double.compare(getSingleClass(target), getSingleClass(o.target));
    }
    
    public static int getSingleClass(double[] target) throws IllegalArgumentException {
        double maxVal = 0;
        int index = 0;
        for (int i = 0; i < target.length; i++) {
            if (target[i] > maxVal) {
                maxVal = target[i];
                index = i;
            }
        }

        return index;
    }
}
