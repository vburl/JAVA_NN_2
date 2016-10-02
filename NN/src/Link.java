public class Link {
    private double weight = 0;	//Link's weight,
    private final Node Source_node;	//Source node
    private final Node Des_node;	//Destination node
    private static int counter = 0;	//Used for ID counting
    private final int ID;	//ID for link
 
    public Link(Node Source, Node Des) {
        Source_node = Source;
        Des_node = Des;
        ID = counter;
        counter++;
    }
 
    public int get_ID(){
    	return ID;
    }
    
    public double get_Weight() {
        return weight;
    }
 
    public void set_Weight(double w) {
        weight = w;
    }
 
    public Node get_Source_node() {
        return Source_node;
    }
 
    public Node get_Des_node() {
        return Des_node;
    }
}