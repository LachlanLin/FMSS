import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Server {
    public void train(int num_iterations) {
        // --- Construct Clients
        for (int u : Data.UTrain)
            Data.client[u] = new Client(u);

        System.out.println("Iter\tPre@5\tRec@5\tF1@5\tNDCG@5\t1-call@5\tAUC");
        for (int iter = 0; iter < num_iterations; iter++) {
            if (iter % 10 == 0) {
                System.out.print(Integer.toString(iter) + "\t");
                Test.test();
            }

            // reset
            Data.SuHashMAP.clear();

            ExecutorService executorService1 = Executors.newFixedThreadPool(8);
            for (int u : Data.UTrain)
                executorService1.submit(Data.client[u]);

            executorService1.shutdown();
            while (true)
                if (executorService1.isTerminated())
                    break;


            ExecutorService executorService2 = Executors.newFixedThreadPool(8);
            for (int u : Data.UTrain)
                executorService2.submit(Data.client[u]);

            executorService2.shutdown();
            while (true)
                if (executorService2.isTerminated())
                    break;


            for (int f = 0; f < Data.d; f++) {
                for (int k = 0; k < Data.d; k++) {
                    Data.S_U[f][k] = 0f;
                }
            }

            for (Integer userID : Data.SuHashMAP.keySet()) {
                for (int f = 0; f < Data.d; f++) {
                    for (int k = 0; k < Data.d; k++) {
                        Data.S_U[f][k] += Data.SuHashMAP.get(userID)[f][k];
                    }
                }
            }

            for (int dimension = 0; dimension < Data.d; dimension++) {

                ExecutorService executorService3 = Executors.newFixedThreadPool(8);
                for (int u : Data.UTrain)
                    executorService3.submit(Data.client[u]);
                // Data.client[u].run();

                executorService3.shutdown();
                while (true)
                    if (executorService3.isTerminated())
                        break;

                // reset
                Data.AHashMap.clear();
                Data.BHashMap.clear();

                ExecutorService executorService4 = Executors.newFixedThreadPool(8);
                for (int u : Data.UTrain)
                    executorService4.submit(Data.client[u]);

                executorService4.shutdown();
                while (true)
                    if (executorService4.isTerminated())
                        break;


                float[] numer = new float[Data.m + 1];

                for (Integer userID : Data.AHashMap.keySet()) {
                    for (Integer itemID : Data.AHashMap.get(userID).keySet()) {
                        numer[itemID] += Data.AHashMap.get(userID).get(itemID);
                    }
                }

                float[] denom = new float[Data.m + 1];
                for (Integer userID : Data.BHashMap.keySet()) {
                    for (Integer itemID : Data.BHashMap.get(userID).keySet()) {
                        denom[itemID] += Data.BHashMap.get(userID).get(itemID);
                    }
                }

                for (int i : Data.U_i.keySet()) {
                    for (int k = 0; k < Data.d; k++) {
                        if (k != dimension) {
                            numer[i] -= Data.c_k * Data.V[i][k] * Data.S_U[dimension][k];
                        }
                    }
                    denom[i] += Data.S_U[dimension][dimension] * Data.c_k + Data.lambda;
                    Data.V[i][dimension] = numer[i] / denom[i];
                }
            }

            for (int f = 0; f < Data.d; f++) {
                for (int k = 0; k <= f; k++) {
                    float val = 0;
                    for (int i = 1; i < Data.m + 1; i++) {
                        val += Data.V[i][f] * Data.V[i][k] * Data.c_k;
                    }
                    Data.S_V[f][k] = val;
                    Data.S_V[k][f] = val;
                }
            }
        }
    }
}
