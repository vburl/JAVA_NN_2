import java.text.*;
import java.util.*;

public class NN {
	static {
		Locale.setDefault(Locale.ENGLISH);
	}

	private boolean isTrained = false;
	private DecimalFormat df;
	
	private ArrayList<Node> inputLayer = new ArrayList<Node>();
	private ArrayList<Node> hiddenLayer = new ArrayList<Node>();
	private ArrayList<Node> outputLayer = new ArrayList<Node>();
	private Node bias = new Node();
	private int[] layers;
	private double epsilon = 0.00000000001;

	private double learningRate = 0.5;
	private double momentum = 0.5;

	// Inputs for xor problem
	private double inputs[][] = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };

	// Corresponding outputs, xor training data
	private double expectedOutputs[][] = { { 0 }, { 1 }, { 1 }, { 0 } };
	private double resultOutputs[][] = { { -1 }, { -1 }, { -1 }, { -1 } }; // dummy init
	private double output[];

	// for weight update all
	private final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();

	public static void main(String[] args) {
		NN nn = new NN(2, 4, 1);
		final int maxRuns = 50000;
		double minErrorCondition = 0.001;
		nn.run(maxRuns, minErrorCondition);
	}

	public NN(int input, int hidden, int output) {
		this.layers = new int[] { input, hidden, output };
		df = new DecimalFormat("#.0#");

		//Create all neurons and connections

		for (int i = 0; i < layers.length; i++) {
			if (i == 0) { // input layer
				for (int j = 0; j < layers[i]; j++) {
					Node neuron = new Node();
					inputLayer.add(neuron);
				}
			} else if (i == 1) { // hidden layer
				for (int j = 0; j < layers[i]; j++) {
					Node neuron = new Node();
					neuron.addInConnectionsS(inputLayer);
					neuron.addBiasConnection(bias);
					hiddenLayer.add(neuron);
				}
			}

			else if (i == 2) { // output layer
				for (int j = 0; j < layers[i]; j++) {
					Node neuron = new Node();
					neuron.addInConnectionsS(hiddenLayer);
					neuron.addBiasConnection(bias);
					outputLayer.add(neuron);
				}
			} else {
				System.out.println("!Error NeuralNetwork init");
			}
		}

		// initialize hidden nodes' link weights
		for (Node neuron : hiddenLayer) {
			ArrayList<Link> connections = neuron.getAllInConnections();
			for (Link conn : connections) {
				double newWeight = getRandom();
				conn.setWeight(newWeight);
			}
		}
		// initialize output nodes' link weights
		for (Node neuron : outputLayer) {
			ArrayList<Link> connections = neuron.getAllInConnections();
			for (Link conn : connections) {
				double newWeight = getRandom();
				conn.setWeight(newWeight);
			}
		}

		// reset id counters
		Node.counter = 0;
		Link.counter = 0;

		if (isTrained) {
			trainedWeights();
			updateAllWeights();
		}
	}

	// random
	double getRandom() {
		return  new Random().nextDouble() * 2 - 1;
	}

	public void setInput(double inputs[]) {
		for (int i = 0; i < inputLayer.size(); i++) {
			inputLayer.get(i).setOutput(inputs[i]);
		}
	}

	public double[] getOutput() {
		double[] outputs = new double[outputLayer.size()];
		for (int i = 0; i < outputLayer.size(); i++)
			outputs[i] = outputLayer.get(i).getOutput();
		return outputs;
	}

	public void activate() {
		for (Node n : hiddenLayer)
			n.calculateOutput();
		for (Node n : outputLayer)
			n.calculateOutput();
	}

	public void BP(double expectedOutput[]) {

		// error check, normalize value ]0;1[
//		for (int i = 0; i < expectedOutput.length; i++) {
//			double d = expectedOutput[i];
//			if (d < 0 || d > 1) {
//				if (d < 0)
//					expectedOutput[i] = 0 + epsilon;
//				else
//					expectedOutput[i] = 1 - epsilon;
//			}
//		}

		int i = 0;
		for (Node n : outputLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				double ak = n.getOutput();
				double ai = con.leftNeuron.getOutput();
				double desiredOutput = expectedOutput[i];

				double partialDerivative = -ak * (1 - ak) * ai * (desiredOutput - ak);
				double deltaWeight = -learningRate * partialDerivative;
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
			i++;
		}

		// update weights for the hidden layer
		for (Node n : hiddenLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				double aj = n.getOutput();
				double ai = con.leftNeuron.getOutput();
				double sumKoutputs = 0;
				int j = 0;
				for (Node out_neu : outputLayer) {
					double wjk = out_neu.getConnection(n.id).getWeight();
					double desiredOutput = (double) expectedOutput[j];
					double ak = out_neu.getOutput();
					j++;
					sumKoutputs = sumKoutputs + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
				}

				double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
				double deltaWeight = -learningRate * partialDerivative;
				double newWeight = con.getWeight() + deltaWeight;
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
		}
	}

	void run(int maxSteps, double minError) {
		int step;
		// Train neural network until minError reached or maxSteps exceeded
		double error = 1.0;
		for (step = 0; step < maxSteps && error > minError; step++) {
			error = 0;
			for (int p = 0; p < inputs.length; p++) {
				setInput(inputs[p]);

				activate();

				output = getOutput();
				resultOutputs[p] = output;

				for (int j = 0; j < expectedOutputs[p].length; j++) {
					error += Math.pow(output[j] - expectedOutputs[p][j], 2);
				}

				BP(expectedOutputs[p]);
			}
		}

		printResult();

		System.out.println("Sum of squared errors = " + error);
		System.out.println("##### EPOCH " + step + "\n");
		if (step == maxSteps) {
			System.out.println("!Error training try again");
		} else {
			printAllWeights();
			printWeightUpdate();
		}
	}

	void printResult() {
		System.out.println("NN example with xor training");
		for (int p = 0; p < inputs.length; p++) {
			System.out.print("INPUTS: ");
			for (int x = 0; x < layers[0]; x++) {
				System.out.print(inputs[p][x] + " ");
			}

			System.out.print("EXPECTED: ");
			for (int x = 0; x < layers[2]; x++) {
				System.out.print(expectedOutputs[p][x] + " ");
			}

			System.out.print("ACTUAL: ");
			for (int x = 0; x < layers[2]; x++) {
				System.out.print(resultOutputs[p][x] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}

	String weightKey(int neuronId, int conId) {
		return "N" + neuronId + "_C" + conId;
	}

	public void updateAllWeights() {
		// update weights for the output layer
		for (Node n : outputLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				String key = weightKey(n.id, con.id);
				double newWeight = weightUpdate.get(key);
				con.setWeight(newWeight);
			}
		}
		// update weights for the hidden layer
		for (Node n : hiddenLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				String key = weightKey(n.id, con.id);
				double newWeight = weightUpdate.get(key);
				con.setWeight(newWeight);
			}
		}
	}

	// trained data
	void trainedWeights() {
		weightUpdate.clear();

		weightUpdate.put(weightKey(3, 0), 1.03);
		weightUpdate.put(weightKey(3, 1), 1.13);
		weightUpdate.put(weightKey(3, 2), -.97);
		weightUpdate.put(weightKey(4, 3), 7.24);
		weightUpdate.put(weightKey(4, 4), -3.71);
		weightUpdate.put(weightKey(4, 5), -.51);
		weightUpdate.put(weightKey(5, 6), -3.28);
		weightUpdate.put(weightKey(5, 7), 7.29);
		weightUpdate.put(weightKey(5, 8), -.05);
		weightUpdate.put(weightKey(6, 9), 5.86);
		weightUpdate.put(weightKey(6, 10), 6.03);
		weightUpdate.put(weightKey(6, 11), .71);
		weightUpdate.put(weightKey(7, 12), 2.19);
		weightUpdate.put(weightKey(7, 13), -8.82);
		weightUpdate.put(weightKey(7, 14), -8.84);
		weightUpdate.put(weightKey(7, 15), 11.81);
		weightUpdate.put(weightKey(7, 16), .44);
	}

	public void printWeightUpdate() {
		System.out.println("printWeightUpdate, put this i trainedWeights() and set isTrained to true");
		// weights for the hidden layer
		for (Node n : hiddenLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				String w = df.format(con.getWeight());
				System.out.println("weightUpdate.put(weightKey(" + n.id + ", " + con.id + "), " + w + ");");
			}
		}
		// weights for the output layer
		for (Node n : outputLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				String w = df.format(con.getWeight());
				System.out.println("weightUpdate.put(weightKey(" + n.id + ", " + con.id + "), " + w + ");");
			}
		}
		System.out.println();
	}

	public void printAllWeights() {
		System.out.println("printAllWeights");
		// weights for the hidden layer
		for (Node n : hiddenLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				double w = con.getWeight();
				System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
			}
		}
		// weights for the output layer
		for (Node n : outputLayer) {
			ArrayList<Link> connections = n.getAllInConnections();
			for (Link con : connections) {
				double w = con.getWeight();
				System.out.println("n=" + n.id + " c=" + con.id + " w=" + w);
			}
		}
		System.out.println();
	}
}