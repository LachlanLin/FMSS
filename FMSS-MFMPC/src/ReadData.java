import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;

public class ReadData {
    public static void readData() throws IOException {
        // ==========================================================
        // --- some statistics, start from index "1"
        Data.userRatingSumTrain = new float[Data.n + 1];
        Data.userRatingNumTrain = new int[Data.n + 1];
        Data.itemRatedSumTrain = new float[Data.m + 1];
        Data.itemRatedNumTrain = new int[Data.m + 1];
        Data.trainSetNum = 0;
        Data.I_u = new HashMap<Integer, HashSet<Integer>>();
        for (int u = 1; u <= Data.n; ++u) {
            Data.I_u.put(u, new HashSet<Integer>());
        }
        Data.I_r_u = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
        for (int u = 1; u <= Data.n; ++u) {
            HashMap<Integer, HashSet<Integer>> tmp = new HashMap<Integer, HashSet<Integer>>();
            Data.I_r_u.put(u, tmp);
            for (int r = 1; r <= Data.p; ++r) {
                Data.I_r_u.get(u).put(r, new HashSet<Integer>());
            }

        }
        // ----------------------------------------------------

        // ==========================================================
        Data.trainUserNo = new HashSet<Integer>();
        Data.I = new HashSet<Integer>();
        for (int i = 1; i <= Data.m; ++i) {
            Data.I.add(i);
        }
        // ----------------------------------------------------

        // ==========================================================
        // --- number of test records
        Data.num_test = 0;
        BufferedReader brTest = new BufferedReader(new FileReader(Data.fnTestData));
        String line = null;
        while ((line = brTest.readLine()) != null) {
            Data.num_test += 1;
        }
        System.out.println("num_test: " + Data.num_test);

        brTest.close();
        // ----------------------------------------------------

        // ==========================================================
        // --- test data
        Data.indexUserTest = new int[Data.num_test];
        Data.indexItemTest = new int[Data.num_test];
        Data.ratingTest = new float[Data.num_test];
        // ----------------------------------------------------

        // ==========================================================
        // Training data: (userID,itemID,rating)
        BufferedReader brTrain = new BufferedReader(new FileReader(Data.fnTrainingData));
        line = null;
        while ((line = brTrain.readLine()) != null) {
            String[] terms = line.split("\\s+|,|;");
            int userID = Integer.parseInt(terms[0]);
            int itemID = Integer.parseInt(terms[1]);
            float rating = Float.parseFloat(terms[2]);
            Data.I_u.get(userID).add(itemID);
            int ratingID = (int) rating;
            // -------------------
            Data.I_r_u.get(userID).get(ratingID).add(itemID);
            // ---------------
            if (Data.trainingDataMap.containsKey(userID)) {
                HashMap<Integer, Float> itemRatingMap = Data.trainingDataMap.get(userID);
                itemRatingMap.put(itemID, rating);
                Data.trainingDataMap.put(userID, itemRatingMap);
            } else {
                HashMap<Integer, Float> itemRatingMap = new HashMap<Integer, Float>();
                itemRatingMap.put(itemID, rating);
                Data.trainingDataMap.put(userID, itemRatingMap);
            }

            // ---
            Data.userRatingSumTrain[userID] += rating;
            Data.userRatingNumTrain[userID] += 1;
            Data.itemRatedSumTrain[itemID] += rating;
            Data.itemRatedNumTrain[itemID] += 1;
            Data.trainUserNo.add(userID);
            Data.trainSetNum += 1;
        }
        brTrain.close();
        System.out.println("Finished reading the training data");
        // ==========================================================

        // ==========================================================
        // Test data: (userID,itemID,rating)
        int id_case = 0; // initialize it to zero
        brTest = new BufferedReader(new FileReader(Data.fnTestData));
        line = null;
        while ((line = brTest.readLine()) != null) {
            String[] terms = line.split("\\s+|,|;");
            int userID = Integer.parseInt(terms[0]);
            int itemID = Integer.parseInt(terms[1]);
            float rating = Float.parseFloat(terms[2]);
            Data.indexUserTest[id_case] = userID;
            Data.indexItemTest[id_case] = itemID;
            Data.ratingTest[id_case] = rating;
            id_case += 1;
        }
        brTest.close();
        System.out.println("Finished reading the test data");
        // ----------------------------------------------------
    }
}
