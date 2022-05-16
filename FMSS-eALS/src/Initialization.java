public class Initialization {
    public static void initialize() {
        // --- model parameters to learn, start from index "1"
        Data.U = new float[Data.n + 1][Data.d];
        Data.V = new float[Data.m + 1][Data.d];
        Data.client = new Client[Data.n + 1];

        Data.S_V = new float[Data.d][Data.d];
        Data.S_U = new float[Data.d][Data.d];

        // ===================================================
        Data.c_k = Data.c_0 / Data.m;

        // Initial U and V
        for (int u = 1; u < Data.n + 1; u++) {
            for (int f = 0; f < Data.d; f++) {
                Data.U[u][f] = (float) ((Math.random() - 0.5) * 0.001);
            }
        }

        for (int i = 1; i < Data.m + 1; i++) {
            for (int f = 0; f < Data.d; f++) {
                Data.V[i][f] = (float) ((Math.random() - 0.5) * 0.001);
            }
        }

        // Initial S_U
        for (int f = 0; f < Data.d; f++) {
            for (int k = 0; k <= f; k++) {
                float val = 0;
                for (int u = 1; u < Data.n + 1; u++) {
                    val += Data.U[u][f] * Data.U[u][k];
                }
                Data.S_U[f][k] = val;
                Data.S_U[k][f] = val;

            }
        }

        // Initial S_V
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
