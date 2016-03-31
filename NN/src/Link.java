public class Link {
    double weight = 0;
    double prevDeltaWeight = 0; // for momentum
    double deltaWeight = 0;
 
    final Node leftNeuron;
    final Node rightNeuron;
    static int counter = 0;
    final public int id; // auto increment, starts at 0
 
    public Link(Node fromN, Node toN) {
        leftNeuron = fromN;
        rightNeuron = toN;
        id = counter;
        counter++;
    }
 
    public double getWeight() {
        return weight;
    }
 
    public void setWeight(double w) {
        weight = w;
    }
 
    public void setDeltaWeight(double w) {
        prevDeltaWeight = deltaWeight;
        deltaWeight = w;
    }
 
    public double getPrevDeltaWeight() {
        return prevDeltaWeight;
    }
 
    public Node get_Source() {
        return leftNeuron;
    }
 
    public Node get_Des() {
        return rightNeuron;
    }
}