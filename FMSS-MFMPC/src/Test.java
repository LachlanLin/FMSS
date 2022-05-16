public class Test {
    public static void test() {
        // --- number of test cases
        float mae = 0;
        float rmse = 0;
        int num = 0;

        // ====================================================
        for (int t = 0; t < Data.num_test; t++) {
            int userID = Data.indexUserTest[t];
            int itemID = Data.indexItemTest[t];
            float rating = Data.ratingTest[t];

            // ===========================================
            // --- prediction via inner product
            float[] U_MPC = new float[Data.d];
            for (int f = 0; f < Data.d; f++) {
                for (int r = 1; r < Data.p + 1; ++r) {
                    int ratingSetSize = Data.I_r_u.get(userID).get(r).size();
                    if (Data.I_r_u.get(userID).get(r).contains(itemID)) {
                        ratingSetSize -= 1;
                    }
                    if (ratingSetSize != 0) {
                        float tmp_sum = 0;
                        for (Integer i_ : Data.I_r_u.get(userID).get(r)) {
                            if (i_ != itemID)
                                tmp_sum += Data.M[i_][r][f];
                        }
                        U_MPC[f] += tmp_sum / (float) Math.sqrt(ratingSetSize);
                    }
                }
            }
            float pred = 0;
            for (int f = 0; f < Data.d; f++) {
                pred += (Data.U[userID][f] + U_MPC[f]) * Data.V[itemID][f];
            }
            pred += Data.b_u[userID] + Data.b_i[itemID] + Data.mu;
            if (pred < Data.MinRating)
                pred = Data.MinRating;
            if (pred > Data.MaxRating)
                pred = Data.MaxRating;

            float err = rating - pred;
            if (!Double.isNaN(err)) {
                mae += Math.abs(err);
                rmse += err * err;
            } else {
                num++;
            }
            // ===========================================
        }
        float MAE = mae / Data.num_test;
        float RMSE = (float) Math.sqrt(rmse / Data.num_test);
        // ====================================================

        // ==========================================================
        // output result
        String result = String.format("%.4f", MAE) + "\t" + String.format("%.4f", RMSE) + "\t" + Integer.toString(num);
        System.out.println(result);
        // ==========================================================
    }
}