import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;

public class ReadData {
    public static void readData() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(Data.fnTrainingData));
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] terms = line.split("\\s+|,|;");
            int userID = Integer.parseInt(terms[0]);
            int itemID = Integer.parseInt(terms[1]);

            // --- add to the whole item set
            Data.ItemSetWhole.add(itemID);
            Data.ITrain.add(itemID);
            Data.UTrain.add(userID);

            // ---
            if (Data.I_u.containsKey(userID)) {
                HashSet<Integer> itemSet = Data.I_u.get(userID);
                itemSet.add(itemID);
                Data.I_u.put(userID, itemSet);
            } else {
                HashSet<Integer> itemSet = new HashSet<Integer>();
                itemSet.add(itemID);
                Data.I_u.put(userID, itemSet);
            }

            // ---
            if (Data.U_i.containsKey(itemID)) {
                HashSet<Integer> userSet = Data.U_i.get(itemID);
                userSet.add(userID);
                Data.U_i.put(itemID, userSet);
            } else {
                HashSet<Integer> userSet = new HashSet<Integer>();
                userSet.add(userID);
                Data.U_i.put(itemID, userSet);
            }

        }
        br.close();

        // ------------------------------
        // === Test data
        if (Data.fnTestData.length() > 0) {
            br = new BufferedReader(new FileReader(Data.fnTestData));
            line = null;
            while ((line = br.readLine()) != null) {
                String[] terms = line.split("\\s+|,|;");
                int userID = Integer.parseInt(terms[0]);
                int itemID = Integer.parseInt(terms[1]);

                // --- add to the whole item set
                Data.ItemSetWhole.add(itemID);

                // ---
                if (Data.TestData.containsKey(userID)) {
                    HashSet<Integer> itemSet = Data.TestData.get(userID);
                    itemSet.add(itemID);
                    Data.TestData.put(userID, itemSet);
                } else {
                    HashSet<Integer> itemSet = new HashSet<Integer>();
                    itemSet.add(itemID);
                    Data.TestData.put(userID, itemSet);
                }
            }
            br.close();
        }
    }
}
