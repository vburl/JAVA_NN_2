import java.util.*;

public class Node {
	private final int ID;
	private static int counter = 0;

	private ArrayList<Link> Links = new ArrayList<Link>();
	private HashMap<Integer, Link> Links_Map = new HashMap<Integer, Link>();
	private Link bias_Link;

	private double node_Output = 0;
	private final double bias = -1;

	public Node() {
		ID = counter;
		counter++;
	}

	public void link_Incoming(ArrayList<Node> incoming_Neuron) {
		for (Node n : incoming_Neuron) {
			Link con = new Link(n, this);
			Links.add(con);
			Links_Map.put(n.get_ID(), con);
		}
	}

	public void calculate_Output() {
		double x = 0;
		for (Link con : Links) {
			double weight = con.get_Weight();
			double input = con.get_Source_node().get_NodeOutput();
			x += (weight * input);
		}
		x += (bias_Link.get_Weight() * bias);

		node_Output = 1.0 / (1.0 + (Math.exp(-x))); // sigmoid function
	}

	public int get_ID() {
		return ID;
	}

	public Link get_Link(int node_ID) {
		return Links_Map.get(node_ID);
	}

	public void add_Bias_link(Node n) {
		Link con = new Link(n, this);
		bias_Link = con;
		Links.add(con);
	}

	public ArrayList<Link> get_Incoming_link() {
		return Links;
	}

	public double get_Bias() {
		return bias;
	}

	public double get_NodeOutput() {
		return node_Output;
	}

	public void set_Output(double o) {
		node_Output = o;
	}
}