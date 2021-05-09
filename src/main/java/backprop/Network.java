package backprop;

public class Network {

    //an array of neuron outputs
    //first index is the layer
    //second index is the neuron number
    public double[][] output;

    //an array of weights
    //first index is the layer
    //second index is the neuron TO WHICH the weight is connected to (on the right side)
    //third index is the neuron FROM WHICH the weight is coming from (on the left side)
    //index 0 isn't used
    private double[][][] weight;

    //an array of biases
    //first index is the layer
    //second index is the neuron number
    //index 0 isn't used
    private double[][] bias;

    //an array of all the deltas calculated for each neuron
    //first index is the layer
    //second index is the neuron
    //index 0 isn't used
    private double[][] delta;

    //beta parameter used in activation functions
    private double beta;

    //whether bipolar activation function is used
    private boolean bipolarActivationFlag;

    //an array of all the output derivatives
    //calculated using unipolarActivationDerivative
    //first index is the layer
    //second index is the neuron
    private double[][] outputDerivative;

    //an array of all the layers sizes
    //number of neurons in each layer
    public final int[] NETWORK_LAYERS_SIZES;

    //size of the input layer
    public final int INPUT_SIZE;

    //size of the output layer
    public final int OUTPUT_SIZE;

    //number of layers
    public final int NETWORK_SIZE;

    public Network(int[] NETWORK_LAYERS_SIZES, double beta, boolean bipolarActivation) {

        this.beta = beta;
        this.bipolarActivationFlag = bipolarActivation;

        this.NETWORK_LAYERS_SIZES = NETWORK_LAYERS_SIZES;
        this.INPUT_SIZE = NETWORK_LAYERS_SIZES[0];
        this.OUTPUT_SIZE = NETWORK_LAYERS_SIZES[NETWORK_LAYERS_SIZES.length - 1];
        this.NETWORK_SIZE = NETWORK_LAYERS_SIZES.length;

        output = new double[NETWORK_SIZE][];
        weight = new double[NETWORK_SIZE][][];
        bias = new double[NETWORK_SIZE][];
        delta = new double[NETWORK_SIZE][];
        outputDerivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            output[i] = new double[NETWORK_LAYERS_SIZES[i]];
            delta[i] = new double[NETWORK_LAYERS_SIZES[i]];
            outputDerivative[i] = new double[NETWORK_LAYERS_SIZES[i]];


            //for the input layer (i = 0) there are no weights from the layer before
            if (i > 0) {

                //initialize the bias array with random values, also creating the array
                bias[i] = NetworkHelper.createRandomArray(NETWORK_LAYERS_SIZES[i], -1, 1);

                //initialize the weights array with random values, also creating the array
                weight[i] = NetworkHelper.createRandomArray(NETWORK_LAYERS_SIZES[i], NETWORK_LAYERS_SIZES[i-1], -1, 1);
            }
        }
    }

    /**
     * Calculates the output of the network.
     * @param input The input array, its size has to be the same as the size of the input layer.
     * @return The output of the network.
     * @throws IllegalArgumentException When the size of the input array != the size of the input layer.
     */
    public double[] calculateOutput(double[] input) throws IllegalArgumentException {
        if (input.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Number of input values:" + input.length + " is not equal the input size:" + INPUT_SIZE);
        }

        //the output of the first layer is actually the input
        this.output[0] = input;

        //iterate through every layer
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            //in each layer, iterate through every neuron
            //neuron - index of the neuron in the current layer
            for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[layer]; neuron++) {

                double sum = 0;

                //for each neuron in previous layer
                //get the weights and outputs between current neuron and the previous neuron and add the product
                //to the sum
                //prevNeuron - index of the neuron in the previous layer
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYERS_SIZES[layer - 1]; prevNeuron++) {
                    sum += weight[layer][neuron][prevNeuron] * output[layer - 1][prevNeuron];
                }

                //add the bias of the neuron to the sum
                sum += bias[layer][neuron];

                //outputs are dependent on the activation function used
                //if unipolar activation function is used
                if (!bipolarActivationFlag) {

                    //calculate the output for each neuron
                    output[layer][neuron] = unipolarActivation(sum, beta);

                    //calculate the derivative output for each neuron
                    //we could use activationDerivative method, but it would mean three times more calculations
                    //as derivative of sigmoid is just sigmoid * (1 - sigmoid), and we are already calculating
                    //our sigmoid over in the output value
                    outputDerivative[layer][neuron] = beta * output[layer][neuron] * (1 - output[layer][neuron]);
                } else {
                    //calculate the output for each neuron
                    output[layer][neuron] = bipolarActivation(sum, beta);

                    //calculate the derivative output for each neuron
                    outputDerivative[layer][neuron] = beta * (1 - output[layer][neuron] * output[layer][neuron]);
                }
            }
        }
        //return the output of the network
        //by returning the last layer's output
        return output[NETWORK_SIZE - 1];
    }

    public void train(double[] input, double[] target, double learningRate) throws IllegalArgumentException {
        //check whether the input and output arrays are valid
        if (input.length != INPUT_SIZE)
            throw new IllegalArgumentException("Input size != input layer size!");

        if (target.length != OUTPUT_SIZE)
            throw new IllegalArgumentException("Output size != output layer size!");

        //set the values of each neuron's outputs given the input
        calculateOutput(input);

        calculateDeltas(target);

        updateWeightsAndBiases(learningRate);

    }

    //calculates the delta value - error multiplied by the derivative of the activaction function
    public void calculateDeltas(double[] target) {
        //calculating delta for the output layer
        for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[NETWORK_SIZE - 1]; neuron++) {
            delta[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) * outputDerivative[NETWORK_SIZE - 1][neuron];
        }

        //calculating delta for the hidden layers, beginning from the back
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[layer]; neuron++) {

                double sum = 0;

                //for every neuron in the next (front) layer, get the delta times weight
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYERS_SIZES[layer + 1]; nextNeuron++) {
                    sum += weight[layer + 1][nextNeuron][neuron] * delta[layer + 1][nextNeuron];
                }

                //calculate the delta for current neuron
                this.delta[layer][neuron] = sum * outputDerivative[layer][neuron];
            }
        }
    }

    //updates the weights depending on the learning rate and the deltas
    public void updateWeightsAndBiases(double learningRate) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[layer]; neuron++) {
                //calculating the delta times learning rate, used in updating both weight and bias
                double deltaTimesLearningRate = - learningRate * delta[layer][neuron];

                //updating the biases
                bias[layer][neuron] += deltaTimesLearningRate;

                //updating weights using previous and current neuron indexes
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYERS_SIZES[layer - 1]; prevNeuron++) {
//                    double deltaWeight = - learningRate * output[layer - 1][prevNeuron] * delta[layer][neuron];
                    weight[layer][neuron][prevNeuron] += deltaTimesLearningRate * output[layer - 1][prevNeuron];
                }
            }
        }
    }

    private static double sigmoid(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    private static double unipolarActivation(double x, double beta) {
        return 1.0 / (1 + Math.exp(-x * beta));
    }

    private static double bipolarActivation(double x, double beta) {
        return Math.tanh(x * beta);
    }

    private static double sigmoidDerivative(double x) {
        return sigmoid(x) * (1.0 - sigmoid(x));
    }


}
