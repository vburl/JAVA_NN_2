import java.text.*;
import java.util.*;

public class NeuralNetwork {
	// Training I/O
	private double training_input[][] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	private double target_output[] = { 0, 1, 1, 0 };
	private double training_output[] = { -1, -1, -1, -1 };

	private int input_nodes = 0;
	private int hidden_nodes = 0;
	private int output_nodes = 0;
	private ArrayList<Node> Input_Layer = new ArrayList<Node>();
	private ArrayList<Node> Hidden_Layer = new ArrayList<Node>();
	private ArrayList<Node> Output_Layer = new ArrayList<Node>();
	private double output;

	private Node Bias_Input = new Node();
	private Node Bias_Hidden = new Node();

	private DecimalFormat df = new DecimalFormat("#0.00000000#");

	private final int max_Batch = 1000000;
	private final double target_error = 0.1;
	private final double learning_rate = 0.05;

	public static void main(String[] args) {
		NeuralNetwork NN = new NeuralNetwork(2, 2, 1);
		NN.Run();
	}

	public NeuralNetwork(int input, int hidden, int output) {
		this.input_nodes = input;
		this.hidden_nodes = hidden;
		this.output_nodes = output;

		// initialize nodes
		Initial_nodes();
		// initialize link weights
		Initial_weight();
	}

	public void Initial_nodes() {
		for (int i = 0; i < input_nodes; i++) {
			Node n = new Node();
			Input_Layer.add(n);
		}
		for (int i = 0; i < hidden_nodes; i++) {
			Node n = new Node();
			n.link_Incoming(Input_Layer);
			n.add_Bias_link(Bias_Input);
			Hidden_Layer.add(n);
		}
		for (int i = 0; i < output_nodes; i++) {
			Node n = new Node();
			n.link_Incoming(Hidden_Layer);
			n.add_Bias_link(Bias_Hidden);
			Output_Layer.add(n);
		}
	}

	public void Initial_weight() {
		for (Node neuron : Hidden_Layer) {
			ArrayList<Link> connections = neuron.get_Incoming_link();
			for (Link conn : connections) {
				double newWeight = new Random().nextDouble() * 2.0 - 1;
				conn.set_Weight(newWeight);
			}
		}
		for (Node neuron : Output_Layer) {
			ArrayList<Link> connections = neuron.get_Incoming_link();
			for (Link conn : connections) {
				double newWeight = new Random().nextDouble() * 2.0 - 1;
				conn.set_Weight(newWeight);
			}
		}
		System.out.println("--------------------Initial weight--------------------");
		print_Weights();
	}

	void Run() {
		int Batch = 0;

		// Train neural network until minError reached or maxSteps exceeded
		double error = 1.0;
		double first_err = 1.0;
		while (Batch < max_Batch && error > target_error) {
			error = Train_model();
			if (Batch == 0) {
				first_err = error;
			}
			Batch++;
		}

		if (Batch == max_Batch) {
			System.out.println("Maximum batch reached! Run again:");
			Initial_weight(); // Re-initial weights
			Run();
		} else {
			System.out.println("First batch error: " + df.format(first_err));
			System.out.println("--------------------Final weight--------------------");
			print_Weights();
			System.out.println("Final error = " + df.format(error));
			System.out.println("Number of batches: " + Batch);
			print_Result();
		}

	}

	public double Train_model() {
		double error = 0;
		for (int training_instance = 0; training_instance < training_input.length; training_instance++) {
			set_Input(training_input[training_instance]);

			feed_Forward();

			BP(target_output[training_instance]);
			
			training_output[training_instance] = output;
			error += Math.pow(output - target_output[training_instance], 2);

		}
		return error;
	}

	public void set_Input(double inputs[]) {
		for (int i = 0; i < Input_Layer.size(); i++) {
			Input_Layer.get(i).set_Output(inputs[i]);
		}
	}
	public void feed_Forward() {
		for (Node n : Hidden_Layer)
			n.calculate_Output();
		for (Node n : Output_Layer)
			n.calculate_Output();
	}

	public void BP(double target) {
		output = Output_Layer.get(0).get_NodeOutput();

		for (Node n : Output_Layer) {
			ArrayList<Link> connections = n.get_Incoming_link();
			for (Link con : connections) {
				double y = n.get_NodeOutput();
				double traversing = con.get_Source_node().get_NodeOutput(); // traversing
				double t = target;

				double D_y = y * (1 - y) * (y - t);
				double d_weight = -learning_rate * D_y * traversing;
				con.set_Weight(con.get_Weight() + d_weight);
			}
		}

		// update weights for the hidden layer
		for (Node n : Hidden_Layer) {
			ArrayList<Link> connections = n.get_Incoming_link();
			for (Link con : connections) {
				double y = n.get_NodeOutput();
				double traversing = con.get_Source_node().get_NodeOutput();
				double sum_K = calculate_Output_Derivative(n, target);

				double D_y = y * (1 - y) * sum_K;
				double d_weight = -learning_rate * D_y * traversing;
				con.set_Weight(con.get_Weight() + d_weight);
			}
		}
	}

	private double calculate_Output_Derivative(Node n, double target) {
		double sum_K = 0;
		for (Node out_node : Output_Layer) {
			double w = out_node.get_Link(n.get_ID()).get_Weight();
			double t_k = target;
			double y_k = out_node.get_NodeOutput();
			sum_K += (y_k - t_k) * y_k * (1 - y_k) * w;
		}
		return sum_K;
	}

	public void print_Weights() {
		System.out.println("Weights:");
		// Hidden layer
		for (Node n : Hidden_Layer) {
			ArrayList<Link> connections = n.get_Incoming_link();
			for (Link con : connections) {
				System.out.println("Node" + con.get_Source_node().get_ID() + " - " + "Node" + con.get_Des_node().get_ID() + " Weight: "
						+ df.format(con.get_Weight()));
			}
		}
		// Output layer
		for (Node n : Output_Layer) {
			ArrayList<Link> connections = n.get_Incoming_link();
			for (Link con : connections) {
				System.out.println("Node" + con.get_Source_node().get_ID() + " - " + "Node" + con.get_Des_node().get_ID() + " Weight: "
						+ df.format(con.get_Weight()));
			}
		}
		System.out.println();
	}

	void print_Result() {
		System.out.println("Training result:");
		for (int training_instance = 0; training_instance < training_input.length; training_instance++) {
			System.out.print("Train" + training_instance + ": ");
			for (int node_idx = 0; node_idx < input_nodes; node_idx++) {
				System.out.print(training_input[training_instance][node_idx] + " ");
				if (node_idx != input_nodes - 1)
					System.out.print("XOR ");
			}
			System.out.print("= ");
			System.out.print(target_output[training_instance]);

			System.out.print(" /Calculated: ");
			System.out.print(df.format(training_output[training_instance]) + "\n");
		}
		System.out.println();
	}
}