package backprop;

public class NetworkHelper {

    /**
     * Creates an array with initial values.
     * @param size Size of the array to create.
     * @param init_value Initial value to set to every index.
     * @return Created array.
     * @throws IllegalArgumentException Thrown when the size is less than 1.
     */
    public static double[] createArray(int size, double init_value) throws IllegalArgumentException {
        if(size < 1){
            throw new IllegalArgumentException("Size is less than 1!");
        }
        double[] ar = new double[size];
        for(int i = 0; i < size; i++){
            ar[i] = init_value;
        }
        return ar;
    }

    /**
     * Creates an array with random values.
     * @param size SIze of the array to create.
     * @param lower_bound Lower bound of the random values.
     * @param upper_bound Upper bound of the random values.
     * @return Created array.
     * @throws IllegalArgumentException Thrown when the size is less than 1.
     */
    public static double[] createRandomArray(int size, double lower_bound, double upper_bound) throws IllegalArgumentException {
        if(size < 1){
            throw new IllegalArgumentException("Size is less than 1!");
        }
        double[] ar = new double[size];
        for(int i = 0; i < size; i++){
            ar[i] = randomValue(lower_bound,upper_bound);
        }
//        double[] ar = new double[size];
//
//        for (int x = 0; x < size; x++) {
//            ar[x] = 0.3;
//        }
        return ar;
    }

    /**
     * Creates a 2d array with random values.
     * @param sizeX Size of the first dimension.
     * @param sizeY Size of the second dimension.
     * @param lower_bound Lower bound of the random values.
     * @param upper_bound Upper bound of the random values.
     * @return Created array.
     * @throws IllegalArgumentException Thrown when the size of any dimension is less than 1.
     */
    public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound) throws IllegalArgumentException {
        if(sizeX < 1 || sizeY < 1){
            throw new IllegalArgumentException("Size X or Y is less than 1!");
        }
        double[][] ar = new double[sizeX][sizeY];
        for(int i = 0; i < sizeX; i++){
            ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
        }

//        double[][] ar = new double[sizeX][sizeY];
//
//        for (int x = 0; x < sizeX; x++) {
//            for (int y = 0; y < sizeY; y++) {
//                ar[x][y] = 0.3;
//            }
//        }

        return ar;
    }

    /**
     * Returns a random value from the bounds.
     * @param lower_bound Lower bound of the random value.
     * @param upper_bound Upper bound of the random value.
     * @return The random value.
     */
    public static double randomValue(double lower_bound, double upper_bound){
        return Math.random()*(upper_bound-lower_bound) + lower_bound;
    }

    /**
     * Returns an array of unique integers.
     * @param lowerBound Lower bound of the random values.
     * @param upperBound Upper bound of the random values.
     * @param amount The size of the array to create.
     * @return Created array.
     * @throws IllegalArgumentException
     */
    public static Integer[] randomValues(int lowerBound, int upperBound, int amount) throws IllegalArgumentException {

        lowerBound --;

        if(amount > (upperBound-lowerBound)){
            throw new IllegalArgumentException("The number of values is bigger than the range given.");
        }

        Integer[] values = new Integer[amount];
        for(int i = 0; i< amount; i++){
            int n = (int)(Math.random() * (upperBound-lowerBound+1) + lowerBound);
            while(containsValue(values, n)){
                n = (int)(Math.random() * (upperBound-lowerBound+1) + lowerBound);
            }
            values[i] = n;
        }
        return values;
    }

    /**
     * Checks whether the array contains the certain value.
     * @param ar Array to check.
     * @param value Value to check for.
     * @param <T>
     * @return True if contains, otherwise False.
     */
    public static <T extends Comparable<T>> boolean containsValue(T[] ar, T value){
        for(int i = 0; i < ar.length; i++){
            if(ar[i] != null){
                if(value.compareTo(ar[i]) == 0){
                    return true;
                }
            }

        }
        return false;
    }

}
