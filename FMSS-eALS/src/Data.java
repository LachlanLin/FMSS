import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

public class Data {

    public static int d = 20;

    public static float c_0 = 800.0f;

    public static float omega = 1.0f;

    public static float lambda = 0.001f;

    public static int rho = 0;
    public static int c = 0;

    public static String fnTrainingData = "";
    public static String fnTestData = "";

    public static int n = 0;
    public static int m = 0;

    public static int num_iterations = 200;

    // === Evaluation
    public static int topK = 5; // top k in evaluation

    // === Purchase Data
    public static HashMap<Integer, HashSet<Integer>> I_u = new HashMap<Integer, HashSet<Integer>>();
    public static HashMap<Integer, HashSet<Integer>> U_i = new HashMap<Integer, HashSet<Integer>>();

    // === Test data
    public static HashMap<Integer, HashSet<Integer>> TestData = new HashMap<Integer, HashSet<Integer>>();

    // ===Whole set of items
    public static HashSet<Integer> ItemSetWhole = new HashSet<Integer>();

    public static HashSet<Integer> ITrain = new HashSet<Integer>();
    public static HashSet<Integer> UTrain = new HashSet<Integer>();

    // === model parameters to learn, start from index "1"
    public static float[][] U;
    public static float[][] V;

    // cache
    public static float[][] S_V;
    public static float[][] S_U;

    public static volatile Client[] client;

    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, Float>> AHashMap = new ConcurrentHashMap<Integer, HashMap<Integer, Float>>();
    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, Float>> BHashMap = new ConcurrentHashMap<Integer, HashMap<Integer, Float>>();
    public static volatile ConcurrentHashMap<Integer, float[][]> SuHashMAP = new ConcurrentHashMap<Integer, float[][]>();

    public static float c_k;

}
