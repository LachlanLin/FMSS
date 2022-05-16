public class Initialization {
    public static void initialization() {
        // --- model parameters to learn, start from index "1"
        Data.U = new float[Data.n + 1][Data.d];
        Data.V = new float[Data.m + 1][Data.d];
        Data.M = new float[Data.m + 1][Data.p + 1][Data.d];
        Data.b_u = new float[Data.n + 1];
        Data.b_i = new float[Data.m + 1];
        Data.mu = 0;

        Data.client = new Client[Data.n + 1];

        Data.countI = new int[Data.m + 1];
        Data.countIR = new int[Data.m + 1][Data.p + 1];

        // Initial U, V, and M
        for (int u = 1; u < Data.n + 1; u++) {
            for (int f = 0; f < Data.d; f++) {
                Data.U[u][f] = (float) ((Math.random() - 0.5) * 0.01);
            }
        }

        for (int i = 1; i < Data.m + 1; i++) {
            for (int f = 0; f < Data.d; f++) {
                Data.V[i][f] = (float) ((Math.random() - 0.5) * 0.01);
            }
        }
        for (int i = 1; i < Data.m + 1; i++) {
            for (int r = 1; r < Data.p + 1; r++) {
                for (int f = 0; f < Data.d; f++) {
                    Data.M[i][r][f] = (float) ((Math.random() - 0.5) * 0.01);
                }
            }
        }
        float sum = 0f;
        for (int u = 1; u < Data.n + 1; u++) {
            sum += Data.userRatingSumTrain[u];
        }
        Data.mu = sum / (float) Data.trainSetNum;
        for (int u = 1; u < Data.n + 1; u++) {
            if (Data.userRatingNumTrain[u] != 0)
                Data.b_u[u] = (Data.userRatingSumTrain[u] - (float) Data.userRatingNumTrain[u] * Data.mu)
                        / (float) Data.userRatingNumTrain[u];
        }
        for (int i = 1; i < Data.m + 1; i++) {
            if (Data.itemRatedNumTrain[i] != 0)
                Data.b_i[i] = (Data.itemRatedSumTrain[i] - (float) Data.itemRatedNumTrain[i] * Data.mu)
                        / (float) Data.itemRatedNumTrain[i];
        }
    }
}
