import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class ReadData {
    public static void readData() throws IOException {
        Data.userRatingNumTrain = new int[Data.n + 1]; // start from index "1"
        Data.itemRatingNumTrain = new int[Data.m + 1]; // start from index "1"
        // ------------------------------

        HashMap<Integer, HashMap<Integer, Integer>> TrainData_time = new HashMap<Integer, HashMap<Integer, Integer>>();
        BufferedReader br = new BufferedReader(new FileReader(Data.fnTrainData));
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] terms = line.split("\\s+|,|;");
            int userID = Integer.parseInt(terms[0]);
            int itemID = Integer.parseInt(terms[1]);
            int timeID = Integer.parseInt(terms[2]);
            Data.ItemSetWhole.add(itemID);

            // --- statistics, used to calculate the performance on different
            // user groups
            Data.userRatingNumTrain[userID] += 1;
            Data.itemRatingNumTrain[itemID] += 1;

            Data.num_train += 1; // the number of total user-item pairs

            // TrainData_time: user-><item,time>
            if (TrainData_time.containsKey(userID)) {
                HashMap<Integer, Integer> itemSet = TrainData_time.get(userID);
                if (itemSet.containsKey(itemID)) {
                    Data.num_train -= 1;
                }
                itemSet.put(itemID, timeID);
                TrainData_time.put(userID, itemSet);
            } else {
                HashMap<Integer, Integer> itemSet = new HashMap<Integer, Integer>();
                itemSet.put(itemID, timeID);
                TrainData_time.put(userID, itemSet);
            }
        }
        br.close();

        //In case that input records are not ordered by time, here we sort them manully.
        for (Integer user : TrainData_time.keySet()) {
            HashMap<Integer, Integer> temp = TrainData_time.get(user);
            ArrayList<Integer> itemlist = order_item(temp);
            Data.TrainData.put(user, itemlist); //TrainData:user->items (time-ordered)
        }

        // ----------------------------------------------------
        if (Data.fnTestData.length() > 0) {
            br = new BufferedReader(new FileReader(Data.fnTestData));
            line = null;
            while ((line = br.readLine()) != null) {
                String[] terms = line.split("\\s+|,|;");
                int userID = Integer.parseInt(terms[0]);
                int itemID = Integer.parseInt(terms[1]);

                if (!Data.TrainData.containsKey(userID)) {
                    continue;
                }

                // ---TestData: user -> items
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

    public static ArrayList<Integer> order_item(HashMap<Integer, Integer> itemset, boolean ascending) {
        // input : HashMap<itemID,timeID>, and return time-ordered ArrayList<itemID>(asending)
        List<Map.Entry<Integer, Integer>> listY = new ArrayList<Map.Entry<Integer, Integer>>(itemset.entrySet());
        listY.sort(new Comparator<Entry<Integer, Integer>>() {
            public int compare(Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) {
                if (ascending) {
                    return o1.getValue().compareTo(o2.getValue());
                } else {
                    return o2.getValue().compareTo(o1.getValue());
                }
            }
        });
        Iterator<Entry<Integer, Integer>> iter = listY.iterator();
        ArrayList<Integer> res = new ArrayList<Integer>();
        while (iter.hasNext()) {
            Map.Entry<Integer, Integer> entry = (Map.Entry<Integer, Integer>) iter.next();
            int itemID = entry.getKey();
            res.add(itemID);
        }
        return res;
    }

    public static ArrayList<Integer> order_item(HashMap<Integer, Integer> itemset) {
        return order_item(itemset, true);
    }
}
