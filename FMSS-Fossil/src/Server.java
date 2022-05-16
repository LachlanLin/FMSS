import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Server {
    public void train() {
        // --- Construct Clients
        for (int u : Data.TrainData.keySet()) {
            Data.client[u] = new Client(u);
        }

        System.out.println("Iter\tPre@5\tRec@5\tF1@5\tNDCG@5\t1-call@5\tAUC");
        for (int iter = 0; iter < Data.num_iterations; iter++) {
            if (iter % 10 == 0) {
                System.out.print(Integer.toString(iter) + "\t");
                Test.test();
            }

            // reset
            Data.V_Gradient.clear();
            Data.W_Gradient.clear();
            Data.bi_Gradient.clear();
            Data.eta_Gradient.clear();
            for (int i = 1; i < Data.m + 1; i++) {
                Data.countOfVAndBi[i] = 0;
            }

            // before secret sharing
            Data.step = 0;
            ExecutorService executorService1 = Executors.newFixedThreadPool(8);
            for (int u : Data.TrainData.keySet())
                executorService1.submit(Data.client[u]);
            executorService1.shutdown();
            while (true)
                if (executorService1.isTerminated())
                    break;

            // after secret sharing
            Data.step = 1;
            ExecutorService executorService2 = Executors.newFixedThreadPool(8);
            for (int u : Data.TrainData.keySet())
                executorService2.submit(Data.client[u]);
            executorService2.shutdown();
            while (true)
                if (executorService2.isTerminated())
                    break;

            for (Integer userID : Data.countOfVAndBiHashMap.keySet()) {
                for (Integer itemID : Data.countOfVAndBiHashMap.get(userID).keySet()) {
                    Data.countOfVAndBi[itemID] += Data.countOfVAndBiHashMap.get(userID).get(itemID);
                }
            }

            if (iter == 0) {
                for (Integer userID : Data.countOfWHashMap.keySet()) {
                    for (Integer itemID : Data.countOfWHashMap.get(userID).keySet()) {
                        Data.countOfW[itemID] += Data.countOfWHashMap.get(userID).get(itemID);
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
            for (int itemID = 1; itemID < Data.m + 1; itemID++) {
                if (Data.countOfVAndBi[itemID] != 0) {
                    for (int f = 0; f < Data.d; f++) {
                        Data.V[itemID][f] -= Data.gamma * grad_V[itemID][f] / (float) (Data.countOfVAndBi[itemID]);
                    }
                }
            }

            // receive all the intermediate b_i gradient
            float[] grad_bi = new float[Data.m + 1];
            for (Integer userID : Data.bi_Gradient.keySet()) {
                for (Integer itemID : Data.bi_Gradient.get(userID).keySet()) {
                    grad_bi[itemID] += Data.bi_Gradient.get(userID).get(itemID);
                }
            }
            for (int itemID = 1; itemID < Data.m + 1; itemID++) {
                if (Data.countOfVAndBi[itemID] != 0) {
                    Data.bi[itemID] -= Data.gamma * grad_bi[itemID] / (float) (Data.countOfVAndBi[itemID]);
                }
            }

            // receive all the intermediate W gradient
            float[][] grad_W = new float[Data.m + 1][Data.d];
            for (Integer userID : Data.W_Gradient.keySet()) {
                for (Integer itemID : Data.W_Gradient.get(userID).keySet()) {
                    float[] tmp_w = Data.W_Gradient.get(userID).get(itemID);
                    for (int f = 0; f < Data.d; f++) {
                        grad_W[itemID][f] += tmp_w[f];
                    }
                }
            }
            for (int itemID = 1; itemID < Data.m + 1; itemID++) {
                if (Data.countOfW[itemID] != 0) {
                    for (int f = 0; f < Data.d; f++) {
                        Data.W[itemID][f] -= Data.gamma * grad_W[itemID][f] / (float) (Data.countOfW[itemID]);
                    }
                }
            }

            // receive all the intermediate eta gradient
            float[] grad_eta = new float[Data.L + 1];
            for (Integer userID : Data.eta_Gradient.keySet()) {
                float[] tmp_eta = Data.eta_Gradient.get(userID);
                for (int l = 1; l < Data.L + 1; l++) {
                    grad_eta[l] += tmp_eta[l];
                }
            }

            int userSize = Data.TrainData.keySet().size();
            for (int l = 1; l < Data.L + 1; l++) {
                Data.eta[l] -= Data.gamma * grad_eta[l] / (float) userSize;
            }

            Data.gamma = Data.gamma * Data.xi; // Decrees $\gamma$
            // =================================================================
        }
    }
}
