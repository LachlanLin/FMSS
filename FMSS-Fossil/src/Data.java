import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

public class Data {
    public static int d = 20;
    // $\alpha$
    public static float alpha = 0.5f;

    // tradeoff $\lambda$
    public static float lambda = 0.01f;

    // learning rate $\gamma$
    public static float gamma = 0.01f;
    public static float xi = 1.0f;

    // the order of Markov Chains
    public static int L = 1;

    public static int step = 0;

    public static int rho = 0;
    public static int c = 0;

    // === Input data files
    public static String fnTrainData = "";
    public static String fnTestData = "";

    public static int n = 0; // number of users
    public static int m = 0; // number of items
    public static int num_train = 0; // number of the total (user, item) pairs in training data
    public static int num_iterations = 0;

    // === Evaluation
    public static int topK = 5; // top k in evaluation
    //
    // === training data
    public static HashMap<Integer, ArrayList<Integer>> TrainData = new HashMap<Integer, ArrayList<Integer>>();

    // === test data
    public static HashMap<Integer, HashSet<Integer>> TestData = new HashMap<Integer, HashSet<Integer>>();
    // === whole data (items)
    public static HashSet<Integer> ItemSetWhole = new HashSet<Integer>();

    // === some statistics, start from index "1"
    public static int[] userRatingNumTrain;
    public static int[] itemRatingNumTrain;

    // private parameter
    public static float[][] eta_u; // l(chain) start from index "1"

    // === model parameters to learn, start from index "1"
    // shared parameters
    public static float[][] W;
    public static float[][] V;
    public static float[] eta; // global parameters, l(chain) start from index "1"
    public static float[] bi; // bias of item

    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, float[]>> V_Gradient = new ConcurrentHashMap<Integer, HashMap<Integer, float[]>>();

    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, float[]>> W_Gradient = new ConcurrentHashMap<Integer, HashMap<Integer, float[]>>();

    public static volatile ConcurrentHashMap<Integer, float[]> eta_Gradient = new ConcurrentHashMap<Integer, float[]>();

    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, Float>> bi_Gradient = new ConcurrentHashMap<Integer, HashMap<Integer, Float>>();

    // clients
    public static volatile Client[] client;

    public static int[] countOfVAndBi;
    public static int[] countOfW;

    public static ConcurrentHashMap<Integer, HashMap<Integer, Integer>> countOfVAndBiHashMap = new ConcurrentHashMap<Integer, HashMap<Integer, Integer>>();
    public static ConcurrentHashMap<Integer, HashMap<Integer, Integer>> countOfWHashMap = new ConcurrentHashMap<Integer, HashMap<Integer, Integer>>();
}
