public class Initialization {

    public static void initialization() {
        // --- model parameters to learn, start from index "1"
        Data.W = new float[Data.m + 1][Data.d];
        Data.V = new float[Data.m + 1][Data.d];
        Data.eta_u = new float[Data.n + 1][Data.L + 1];
        Data.eta = new float[Data.L + 1];
        Data.bi = new float[Data.m + 1]; // bias of item
        Data.countOfVAndBi = new int[Data.m + 1];
        Data.countOfW = new int[Data.m + 1];
        Data.client = new Client[Data.n + 1];

        // --- initialization of W, V, eta_u, and eta
        for (int i = 1; i < Data.m + 1; i++) {
            for (int f = 0; f < Data.d; f++) {
                Data.W[i][f] = (float) ((Math.random() - 0.5) * 0.01);
                Data.V[i][f] = (float) ((Math.random() - 0.5) * 0.01);
            }
        }

        for (int u = 1; u <= Data.n; u++) {
            for (int l = 1; l < Data.L + 1; l++) {
                Data.eta_u[u][l] = (float) ((Math.random() - 0.5) * 0.01);
            }
        }
        for (int l = 1; l <= Data.L; l++) {
            Data.eta[l] = (float) ((Math.random() - 0.5) * 0.01);
        }
        // ======================================================
        // --- initialization of bi
        float g_avg = 0;
        for (int i = 1; i < Data.m + 1; i++) {
            g_avg += Data.itemRatingNumTrain[i];
        }
        g_avg = g_avg / Data.n / Data.m;
        System.out.println("The global average rating:" + Float.toString(g_avg));

        for (int i = 1; i < Data.m + 1; i++) {
            Data.bi[i] = (float) Data.itemRatingNumTrain[i] / Data.n - g_avg;
        }
    }
}
