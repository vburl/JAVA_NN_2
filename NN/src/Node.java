import java.util.*;
 
public class Node {   
    static int counter = 0;
    final public int id;  // auto increment, starts at 0
    Link biasConnection;
    final double bias = -1;
    double output;
     
    ArrayList<Link> Inconnections = new ArrayList<Link>();
    HashMap<Integer,Link> connectionLookup = new HashMap<Integer,Link>();
     
    public Node(){        
        id = counter;
        counter++;
    }
     
    /**
     * Compute Sj = Wij*Aij + w0j*bias
     */
    public void calculateOutput(){
        double s = 0;
        for(Link con : Inconnections){
            Node leftNeuron = con.get_Source();
            double weight = con.getWeight();
            double a = leftNeuron.getOutput();
             
            s = s + (weight*a);
        }
        s = s + (biasConnection.getWeight()*bias);
         
        output = 1.0 / (1.0 +  (Math.exp(-s))); //Sigmoid function
    }
     
    public void addInConnectionsS(ArrayList<Node> inNeurons){
        for(Node n: inNeurons){
            Link con = new Link(n,this);
            Inconnections.add(con);
            connectionLookup.put(n.id, con);
        }
    }
     
    public Link getConnection(int neuronIndex){
        return connectionLookup.get(neuronIndex);
    }
 
    public void addInConnection(Link con){
        Inconnections.add(con);
    }
    public void addBiasConnection(Node n){
        Link con = new Link(n,this);
        biasConnection = con;
        Inconnections.add(con);
    }
    public ArrayList<Link> getAllInConnections(){
        return Inconnections;
    }
     
    public double getBias() {
        return bias;
    }
    public double getOutput() {
        return output;
    }
    public void setOutput(double o){
        output = o;
    }
}