package backprop;

public class Network {

    /**
     * Tablica sygnałów wyjściowych neuronów.
     * Indeks pierwszy jest warstwą.
     * Indeks drugi jest indeksem neuronu.
     */
    public double[][] output;

    /**
     * Tablica współczynników wagowych.
     * Indeks pierwszy jest warstwą.
     * Indeks drugi jest indeksem neuronu w warstwie, do którego wejścia skierowana jest waga.
     * Indeks trzeci jest indeksem neuronu w warstwie poprzedniej, od którego wyjścia wychodzi waga.
     */
    private double[][][] weight;

    /**
     * Tablica biasów (przesunięć).
     * Pierwszy indeks jest warstwą.
     * Drugi indeks jest indeksem neuronu z danej warstwy.
     * Indeks 0 nie jest wykorzysywany.
     */
    private double[][] bias;

    /**
     * Tablica wartości błędów (delt) obliczonych dla każdego z neuronów.
     * Pierwszy indeks to warstwa.
     * Drugi indeks to indeks neuronu.
     * Indeks 0 nie jest wykorzystywany.
     */
    private double[][] delta;

    /**
     * Parametr Beta wykorzystywany w funkcjach aktywacji - współczynnik odpowiadający za stromość funkcji.
     */
    private double beta;

    /**
     * Flaga informująca o tym, czy do obliczeń powinna być wykorzystywana bipolarna funkcja aktywacji.
     */
    private boolean bipolarActivationFlag;

    /**
     * Tablica pochodnych wyjść neuronów.
     * Wykorzystywana do obliczania delt.
     * Pierwszy indeks jest warstwą.
     * Drugi indeks jest indeksem neuronu.
     */
    private double[][] outputDerivative;

    /**
     * Tablica zawierająca liczbę neuronów w każdej z warstw.
     */
    public final int[] NETWORK_LAYERS_SIZES;

    /**
     * Liczba wejść.
     */
    public final int INPUT_SIZE;

    /**
     * Liczba neuronów w warstwie wyjściowej.
     */
    public final int OUTPUT_SIZE;

    /**
     * Liczba warstw.
     */
    public final int NETWORK_SIZE;

    public Network(int[] NETWORK_LAYERS_SIZES, double beta, boolean bipolarActivation) {

        this.beta = beta;
        this.bipolarActivationFlag = bipolarActivation;

        this.NETWORK_LAYERS_SIZES = NETWORK_LAYERS_SIZES;
        this.INPUT_SIZE = NETWORK_LAYERS_SIZES[0];
        this.OUTPUT_SIZE = NETWORK_LAYERS_SIZES[NETWORK_LAYERS_SIZES.length - 1];
        this.NETWORK_SIZE = NETWORK_LAYERS_SIZES.length;

        // Tworzenie tablic dla pól klasy.
        output = new double[NETWORK_SIZE][];
        weight = new double[NETWORK_SIZE][][];
        bias = new double[NETWORK_SIZE][];
        delta = new double[NETWORK_SIZE][];
        outputDerivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            output[i] = new double[NETWORK_LAYERS_SIZES[i]];
            delta[i] = new double[NETWORK_LAYERS_SIZES[i]];
            outputDerivative[i] = new double[NETWORK_LAYERS_SIZES[i]];

            // Dla i = 0 warstwą wejściową jest wejście sieci.
            if (i > 0) {
                // Zakres, zapewniający optymalną inicjalizację współczynników wagowych,
                // Zgodnie z rozważaniami Nguyena - Widrowa.
                // Liczba neuronów w warstwie do potęgi 1 / rozmiar wejścia sieci.
                double bound = Math.pow(NETWORK_LAYERS_SIZES[i], 1.0 / INPUT_SIZE);

                // Inicjalizacja ostatniej wagi, której rozważania Nguyena - Widrowa nie dotyczą.
                if (i == NETWORK_SIZE - 1) {
                    bias[i] = NetworkHelper.createRandomArray(NETWORK_LAYERS_SIZES[i], -0.5, 0.5);
                    weight[i] = NetworkHelper.createRandomArray(NETWORK_LAYERS_SIZES[i], NETWORK_LAYERS_SIZES[i-1], -0.5, 0.5);
                } else {
                    bias[i] = NetworkHelper.createRandomArray(NETWORK_LAYERS_SIZES[i], -bound, bound);
                    weight[i] = NetworkHelper.createRandomArray(NETWORK_LAYERS_SIZES[i], NETWORK_LAYERS_SIZES[i-1], -bound, bound);
                }
            }
        }
    }

    /**
     * Uczy sieć jednokrotnie wykorzystując jedną parę uczącą.
     * @param input Tablica reprezentująca wejście sieci.
     * @param target Tablica reprezentująca oczekiwane wyjście sieci.
     * @param learningRate Współczynnik uczenia sieci.
     * @throws IllegalArgumentException Gdy rozmiary wejścia lub wyjścia sieci nie są zgodne z oczekiwanymi.
     */
    public void train(double[] input, double[] target, double learningRate) throws IllegalArgumentException {
        if (input.length != INPUT_SIZE)
            throw new IllegalArgumentException("Input size != input layer size!");

        if (target.length != OUTPUT_SIZE)
            throw new IllegalArgumentException("Output size != output layer size!");

        // Oblicza wyjście dla każdego neuronu w sieci na podstawie wejścia.
        calculateOutput(input);

        // Oblicza delty dla każdego neuronu w sieci, wykonując wsteczną propagację błędu.
        calculateDeltas(target);

        // Aktualizuje współczynniki wagowe oraz biasy dla każdego neuronu w sieci.
        updateWeightsAndBiases(learningRate);
    }

    /**
     * Oblicza wyjście sieci.
     * @param input Tablica wejść sieci.
     * @return Tablica wyjść sieci.
     * @throws IllegalArgumentException Gdy rozmiar tablicy wejściowej != rozmiar wejścia sieci.
     */
    public double[] calculateOutput(double[] input) throws IllegalArgumentException {
        if (input.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Number of input values:" + input.length + " is not equal the input size:" + INPUT_SIZE);
        }

        // Wyjściem pierwszej warstwy w kodzie jest w rzeczywistości tablica wejść sieci.
        this.output[0] = input;

        // Iteracja przez każdą z warstw.
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            // W każdej z warstw iteracja przez każdy z neuronów.
            // neuron - indeks neuronu w .
            for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[layer]; neuron++) {

                double sum = 0;

                // Dla każdego neuronu z warstwy poprzedniej, oblicz iloczyn wagi oraz wyjścia poprzedniego neuronu,
                // po czym dodaj ją do sumy.
                // prevNeuron - indeks neuronu z poprzedniej warstwy.
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYERS_SIZES[layer - 1]; prevNeuron++) {
                    sum += weight[layer][neuron][prevNeuron] * output[layer - 1][prevNeuron];
                }

                // Dodaj bias do sumy.
                sum += bias[layer][neuron];

                // Oblicz wyjście neuronu w zależności od wykorzysywanej funkcji aktywacji.
                if (!bipolarActivationFlag) {

                    // Oblicz wyjście neuronu wykorzystując unipolarną funkcję aktywacji.
                    output[layer][neuron] = unipolarActivation(sum, beta);

                    // Oblicz pochodną wyjścia neuronu wykorzystując unipolarną funkcję aktywacji.
                    outputDerivative[layer][neuron] = beta * output[layer][neuron] * (1 - output[layer][neuron]);
                } else {
                    // Oblicz wyjście neuronu wykorzystując bipolarną funkcję aktywacji.
                    output[layer][neuron] = bipolarActivation(sum, beta);

                    // Oblicz pochodną wyjścia neuronu wykorzystując bipolarną funkcję aktywacji.
                    outputDerivative[layer][neuron] = beta * (1 - output[layer][neuron] * output[layer][neuron]);
                }
            }
        }
        // Zwróć tablicę reprezentującą wyjście sieci.
        return output[NETWORK_SIZE - 1];
    }

    /**
     * Oblicza deltę dla każdego z neuronów. Implementuje algorytm wstecznej propagacji,
     * obliczając wartości od ostatniej warstwy sieci.
     * @param target Tablica reprezentująca oczekiwane wyjście sieci.
     */
    public void calculateDeltas(double[] target) {
        // Obliczanie delty dla ostatniej warstwy sieci.
        for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[NETWORK_SIZE - 1]; neuron++) {
            delta[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) * outputDerivative[NETWORK_SIZE - 1][neuron];
        }

        // Obliczanie delty dla warstw ukrtych, w kierunku wstecznym.
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[layer]; neuron++) {
                double sum = 0;

                // Dla każdego neuronu z następnej warstwy, dodaj do sumy jego deltę
                // pomnożoną razy wagę łączącą dany neuron z neuronem z warstwy następnej.
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYERS_SIZES[layer + 1]; nextNeuron++) {
                    sum += weight[layer + 1][nextNeuron][neuron] * delta[layer + 1][nextNeuron];
                }

                // Oblicz deltę dla danego neuronu.
                this.delta[layer][neuron] = sum * outputDerivative[layer][neuron];
            }
        }
    }

    /**
     * Aktualizuje współczynniki wagowe oraz biasy (przesunięcia) każdego neuronu w sieci,
     * wykorzystując współczynnik uczenia sieci.
     * @param learningRate Współczynnik uczenia sieci.
     */
    public void updateWeightsAndBiases(double learningRate) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYERS_SIZES[layer]; neuron++) {
                //calculating the delta times learning rate, used in updating both weight and bias
                // Dla każdego neuronu oblicza wartość delty pomnożonej przez ujemny współczynnik uczenia.
                double deltaTimesLearningRate = - learningRate * delta[layer][neuron];

                // Aktualizacja biasów.
                bias[layer][neuron] += deltaTimesLearningRate;

                // Aktualizuje wagi, wykorzystując wcześniej obliczony iloczyn pomnożony przez wyjście neuronu
                // z warstwy poprzedniej.
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYERS_SIZES[layer - 1]; prevNeuron++) {
                    weight[layer][neuron][prevNeuron] += deltaTimesLearningRate * output[layer - 1][prevNeuron];
                }
            }
        }
    }

    /**
     * Unipolarna funkcja aktywacji neuronu.
     * @param x Parametr wejściowy funkcji.
     * @param beta Współczynnik beta funkcji aktywacji.
     * @return Wyjście funkcji aktywacji.
     */
    private static double unipolarActivation(double x, double beta) {
        return 1.0 / (1 + Math.exp(-x * beta));
    }

    /**
     * Bipolarna funkcja aktywacji neuronu.
     * @param x Parametr wejściowy funkcji.
     * @param beta Współczynnik beta funkcji aktywacji.
     * @return Wyjście funkcji aktywacji.
     */
    private static double bipolarActivation(double x, double beta) {
        return Math.tanh(x * beta);
    }
}
