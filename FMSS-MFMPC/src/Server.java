import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Server {
    public void train(int num_iterations) {
        // --- Construct Clients
        for (int u : Data.trainUserNo)
            Data.client[u] = new Client(u);

        // --- Train
        System.out.println("Iter\tMAE\tRMSE\tNaN");
        for (int iter = 0; iter < num_iterations; iter++) {
            if (iter % 10 == 0) {
                System.out.print(Integer.toString(iter) + "\t");
                Test.test();
            }

            // reset
            Data.V_Gradient.clear();
            Data.M_Gradient.clear();
            Data.b_i_Gradient.clear();
            Data.mu_Gradient.clear();

            // before secret sharing
            Data.step = 0;
            ExecutorService executorService1 = Executors.newFixedThreadPool(8);
            for (int u : Data.trainUserNo)
                executorService1.submit(Data.client[u]);
            executorService1.shutdown();
            while (true)
                if (executorService1.isTerminated())
                    break;

            // after secret sharing
            Data.step = 1;
            ExecutorService executorService2 = Executors.newFixedThreadPool(8);
            for (int u : Data.trainUserNo)
                executorService2.submit(Data.client[u]);
            executorService2.shutdown();
            while (true)
                if (executorService2.isTerminated())
                    break;

            if (iter == 0) {
                for (Integer userID : Data.countIHashMap.keySet()) {
                    for (Integer itemID : Data.countIHashMap.get(userID).keySet()) {
                        Data.countI[itemID] += Data.countIHashMap.get(userID).get(itemID);
                    }
                }
                for (Integer userID : Data.countIRHashMap.keySet()) {
                    for (Integer itemID : Data.countIRHashMap.get(userID).keySet()) {
                        for (Integer ratingID : Data.countIRHashMap.get(userID).get(itemID).keySet()) {
                            Data.countIR[itemID][ratingID] += Data.countIRHashMap.get(userID).get(itemID).get(ratingID);
                        }
                    }
                }
            }

            // receive all the intermediate V gradient
            float[][] grad_V = new float[Data.m + 1][Data.d];
            for (Integer userID : Data.V_Gradient.keySet()) {
                for (Integer itemID : Data.V_Gradient.get(userID).keySet()) {
                    float[] tmp_v = Data.V_Gradient.get(userID).get(itemID);
                    for (int f = 0; f < Data.d; f++) {
                        grad_V[itemID][f] += tmp_v[f];
                    }
                }
            }

            // receive all the intermediate M gradient
            float[][][] grad_M = new float[Data.m + 1][Data.p + 1][Data.d];
            for (Integer userID : Data.M_Gradient.keySet()) {
                for (Integer itemID : Data.M_Gradient.get(userID).keySet()) {
                    for (Integer ratingID : Data.M_Gradient.get(userID).get(itemID).keySet()) {
                        float[] tmp_m = Data.M_Gradient.get(userID).get(itemID).get(ratingID);
                        for (int f = 0; f < Data.d; f++) {
                            grad_M[itemID][ratingID][f] += tmp_m[f];
                        }
                    }
                }
            }

            // receive all the intermediate b_i gradient
            float[] grad_b_i = new float[Data.m + 1];
            for (Integer userID : Data.b_i_Gradient.keySet()) {
                for (Integer itemID : Data.b_i_Gradient.get(userID).keySet()) {
                    grad_b_i[itemID] += Data.b_i_Gradient.get(userID).get(itemID);
                }
            }

            // receive all the intermediate mu gradient
            float grad_mu = 0;
            for (Integer userID : Data.mu_Gradient.keySet()) {
                grad_mu += Data.mu_Gradient.get(userID);
            }

            // Update V
            for (int itemID = 1; itemID < Data.m + 1; itemID++) {
                if (Data.countI[itemID] != 0) {
                    for (int f = 0; f < Data.d; f++) {
                        Data.V[itemID][f] -= Data.gamma * grad_V[itemID][f] / (float) (Data.countI[itemID]);
                    }
                }
            }

            // Update M
            for (int itemID = 1; itemID < Data.m + 1; itemID++) {
                for (int ratingID = 1; ratingID < Data.p + 1; ratingID++) {
                    if (Data.countIR[itemID][ratingID] != 0) {
                        for (int f = 0; f < Data.d; f++) {
                            Data.M[itemID][ratingID][f] -= Data.gamma * grad_M[itemID][ratingID][f]
                                    / (float) (Data.countIR[itemID][ratingID]);
                        }
                    }
                }
            }

            // Update b_i
            for (int itemID = 1; itemID < Data.m + 1; itemID++) {
                if (Data.countI[itemID] != 0) {
                    Data.b_i[itemID] -= Data.gamma * grad_b_i[itemID] / (float) (Data.countI[itemID]);
                }
            }

            // Update mu
            Data.mu -= Data.gamma * grad_mu / (float) (Data.trainSetNum);

            Data.gamma = Data.gamma * Data.xi; // Decrees $\gamma$
        }
    }
}