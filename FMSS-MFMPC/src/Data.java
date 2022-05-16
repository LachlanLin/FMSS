import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

public class Data {
    // === Configurations
    // the number of latent dimensions
    public static int d = 20;

    // tradeoff $\lambda$
    public static float lambda = 0.01f;

    // learning rate $\gamma$
    public static float gamma = 0.01f;
    public static float xi = 1.0f;

    public static int step = 0;

    public static int rho = 0;
    public static int c = 0;

    // === Input data files
    public static String fnTrainingData = "";
    public static String fnTestData = "";

    public static int n = 943; // number of users
    public static int m = 1682; // number of items
    public static int p = 5; // range of ratings
    public static int num_test; // number of test triples of (user,item,rating)
    public static float MinRating = 1.0f; // minimum rating value
    public static float MaxRating = 5.0f; // maximum rating value

    // scan number over the whole data
    public static int num_iterations = 500;

    // === training data
    public static HashSet<Integer> trainUserNo;
    // training tuple (user id, (item id, rating value))
    public static HashMap<Integer, HashMap<Integer, Float>> trainingDataMap = new HashMap<Integer, HashMap<Integer, Float>>();

    // === test data
    public static int[] indexUserTest;
    public static int[] indexItemTest;
    public static float[] ratingTest;

    // === some statistics, start from index "1"
    public static float[] userRatingSumTrain;
    public static int[] userRatingNumTrain;
    public static float[] itemRatedSumTrain;
    public static int[] itemRatedNumTrain;
    public static int trainSetNum;
    // (userID, (ratingID, itemID))
    public static HashMap<Integer, HashMap<Integer, HashSet<Integer>>> I_r_u;
    public static HashMap<Integer, HashSet<Integer>> I_u; // item set rated by user u

    public static HashSet<Integer> I; // item set

    // === model parameters to learn, start from index "1"
    public static float[][] U;
    public static float[][] V;
    public static float[] b_u;
    public static float[] b_i;
    public static float mu;
    public static float[][][] M; // (item, rating, dimension)

    // intermediate gradient, (userID, (itemID, gradient))
    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, float[]>> V_Gradient = new ConcurrentHashMap<Integer, HashMap<Integer, float[]>>();

    // intermediate gradient, (userID, (itemID, (ratingID, gradient)))
    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, HashMap<Integer, float[]>>> M_Gradient = new ConcurrentHashMap<Integer, HashMap<Integer, HashMap<Integer, float[]>>>();

    // intermediate gradient, (userID, (itemID, gradient))
    public static volatile ConcurrentHashMap<Integer, HashMap<Integer, Float>> b_i_Gradient = new ConcurrentHashMap<Integer, HashMap<Integer, Float>>();

    // intermediate gradient, (userID, gradient)
    public static volatile ConcurrentHashMap<Integer, Float> mu_Gradient = new ConcurrentHashMap<Integer, Float>();

    // clients
    public static volatile Client[] client;

    public static int[] countI;
    public static int[][] countIR;

    public static ConcurrentHashMap<Integer, HashMap<Integer, Integer>> countIHashMap = new ConcurrentHashMap<Integer, HashMap<Integer, Integer>>();
    public static ConcurrentHashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>> countIRHashMap = new ConcurrentHashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>();
}