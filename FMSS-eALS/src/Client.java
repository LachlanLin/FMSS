import java.util.LinkedList;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;

public class Client implements Runnable {
    private int No; // No of Client u
    private int step; // iteration number

    private List<Integer> I_u_sample; // sample item set of client u
    private List<Integer> I_u; // rated item set of client u
    private HashSet<Integer> clientsToSent;

    // intermediate value sent to the server
    private float[][] S_U;
    private HashMap<Integer, Float> A;
    private HashMap<Integer, Float> B;

    // received intermediate values from other clients
    private volatile HashMap<Integer, HashMap<Integer, Float>> receivedA;
    private volatile HashMap<Integer, HashMap<Integer, Float>> receivedB;
    private volatile HashMap<Integer, float[][]> receivedSu;

    // initialization of client u
    Client(int userID) {
        No = userID;
        step = 0;

        I_u = new LinkedList<Integer>(Data.I_u.get(this.No));
        I_u_sample = new LinkedList<Integer>(Data.ITrain);
        I_u_sample.removeAll(this.I_u);
        clientsToSent = new HashSet<Integer>();

        S_U = new float[Data.d][Data.d];
        A = new HashMap<Integer, Float>();
        B = new HashMap<Integer, Float>();

        receivedA = new HashMap<Integer, HashMap<Integer, Float>>();
        receivedB = new HashMap<Integer, HashMap<Integer, Float>>();
        receivedSu = new HashMap<Integer, float[][]>();
    }

    @Override
    public void run() {
        if (step == 0) {
            step0();
        } else if (step == 1) {
            step1();
        } else if (step > 1 && step < 2 + Data.d * 2) {
            int f = (step - 2) / 2;
            if (step % 2 == 0) {
                step2(f);
            } else {
                step3(f);
            }
        } else {
            System.out.println("ERROR");
        }
        step = (step + 1) % (2 + Data.d * 2);
    }

    private void step0() {
        for (int f = 0; f < Data.d; f++) {
            for (int k = 0; k < Data.d; k++) {
                this.S_U[f][k] = 0f;
            }
        }

        this.clientsToSent.clear();

        HashMap<Integer, Float> prediction = new HashMap<Integer, Float>();
        for (int i : this.I_u) {
            float r_ui = 0;
            for (int f = 0; f < Data.d; f++) {
                r_ui += Data.U[this.No][f] * Data.V[i][f];
            }
            prediction.put(i, r_ui);
        }

        for (int f = 0; f < Data.d; f++) {
            float numer = 0, denom = 0;
            for (int k = 0; k < Data.d; k++) {
                if (k != f) {
                    numer -= Data.U[this.No][k] * Data.S_V[f][k];
                }
            }
            for (int i : this.I_u) {
                prediction.put(i, prediction.get(i) - Data.U[this.No][f] * Data.V[i][f]);
                numer += (Data.omega * 1 - (Data.omega - Data.c_k) * prediction.get(i)) * Data.V[i][f];
                denom += (Data.omega - Data.c_k) * Data.V[i][f] * Data.V[i][f];
            }
            denom += Data.S_V[f][f] + Data.lambda;
            // update user factor
            Data.U[this.No][f] = numer / denom;

            // update the prediction cache
            for (int i : Data.I_u.get(this.No)) {
                prediction.put(i, prediction.get(i) + Data.U[this.No][f] * Data.V[i][f]);
            }

        }

        // update S_U
        for (int f = 0; f < Data.d; f++) {
            for (int k = 0; k <= f; k++) {
                float val1 = Data.U[this.No][f] * Data.U[this.No][k];
                this.S_U[f][k] = val1;
                this.S_U[k][f] = val1;
            }
        }

        // sample some clients to sent random numbers, i.e., \mathcal{N}_u
        while (this.clientsToSent.size() < Data.c) {
            int userID = (int) (Data.n * Math.random()) + 1;
            if (Data.UTrain.contains(userID)) {
                this.clientsToSent.add(userID);
            }
        }

        // generate random numbers of S_U and send them to other clients
        for (Integer userID : this.clientsToSent) {
            float[][] sent_SU = new float[Data.d][Data.d];
            for (int f = 0; f < Data.d; f++) {
                for (int k = 0; k < Data.d; k++) {
                    float randomSU = (float) Math.random() * 3.0f;
                    sent_SU[f][k] = randomSU;
                    this.S_U[f][k] -= randomSU;
                }
            }
            Data.client[userID].received_Su(this.No, sent_SU);
        }
    }

    private void step1() {
        // receive the S_U, calculate the sum of them, and send them to the server
        for (Integer userID : receivedSu.keySet()) {
            float[][] tmp_su = receivedSu.get(userID);
            for (int f = 0; f < Data.d; f++) {
                for (int k = 0; k < Data.d; k++) {
                    this.S_U[f][k] += tmp_su[f][k];
                }
            }
        }
        Data.SuHashMAP.put(this.No, this.S_U);
        receivedSu.clear();
    }

    private void step2(int dimension) {
        A.clear();
        B.clear();

        this.clientsToSent.clear();
        for (int i : this.I_u) {
            float r_ui = 0;
            for (int f = 0; f < Data.d; f++) {
                r_ui += Data.U[this.No][f] * Data.V[i][f];
            }

            float tmp_A = (Data.omega * 1
                    - (Data.omega - Data.c_k) * (r_ui - Data.U[this.No][dimension] * Data.V[i][dimension]))
                    * Data.U[this.No][dimension];
            this.A.put(i, tmp_A);

            float tmp_B = (Data.omega - Data.c_k) * Data.U[this.No][dimension] * Data.U[this.No][dimension];
            this.B.put(i, tmp_B);
        }

        // sample fake items
        int sample_number = Math.min(Data.rho * this.I_u.size(), I_u_sample.size());
        HashSet<Integer> fakeItemSet = new HashSet<Integer>();
        while (fakeItemSet.size() < sample_number) {
            int randomIndex = (int) (Math.random() * this.I_u_sample.size());
            int randomItem = this.I_u_sample.get(randomIndex);
            if (Data.ITrain.contains(randomItem)) {
                fakeItemSet.add(randomItem);
            }
        }
        for (Integer i : fakeItemSet) {
            A.put(i, 0f);
            B.put(i, 0f);
        }
        // sample some clients to sent random numbers, i.e., \mathcal{N}_u
        while (this.clientsToSent.size() < Data.c) {
            int userID = (int) (Data.n * Math.random()) + 1;
            if (Data.UTrain.contains(userID)) {
                this.clientsToSent.add(userID);
            }
        }

        // generate random numbers and send them to other clients
        for (Integer userID : this.clientsToSent) {
            // generate random numbers of A and send them to others.
            HashMap<Integer, Float> ASent = new HashMap<Integer, Float>();
            for (Integer itemID : this.A.keySet()) {
                float randomNum = (float) Math.random() * 3.0f;
                ASent.put(itemID, randomNum);
                A.put(itemID, A.get(itemID) - randomNum);
            }
            Data.client[userID].received_A(this.No, ASent);

            // generate random numbers of B and send them to others.
            HashMap<Integer, Float> BSent = new HashMap<Integer, Float>();
            for (Integer itemID : this.B.keySet()) {
                float randomNum = (float) Math.random() * 3.0f;
                BSent.put(itemID, randomNum);
                B.put(itemID, B.get(itemID) - randomNum);
            }
            Data.client[userID].received_B(this.No, BSent);
        }
    }

    private void step3(int dimension) {
        // receive the A, calculate the sum of them, and send them to the server
        for (Integer userID : receivedA.keySet()) {
            for (Integer itemID : receivedA.get(userID).keySet()) {
                if (this.A.containsKey(itemID)) {
                    A.put(itemID, (float) A.get(itemID) + receivedA.get(userID).get(itemID));
                } else {
                    A.put(itemID, (float) receivedA.get(userID).get(itemID));
                }
            }
        }
        Data.AHashMap.put(this.No, this.A);
        // receive the B, calculate the sum of them, and send them to the server
        for (Integer userID : receivedB.keySet()) {
            for (Integer itemID : receivedB.get(userID).keySet()) {
                if (this.B.containsKey(itemID)) {
                    B.put(itemID, (float) B.get(itemID) + receivedB.get(userID).get(itemID));
                } else {
                    B.put(itemID, (float) receivedB.get(userID).get(itemID));
                }
            }
        }
        Data.BHashMap.put(this.No, this.B);
        receivedA.clear();
        receivedB.clear();
    }

    private synchronized void received_Su(int senderID, float[][] Su) {
        receivedSu.put(senderID, Su);
    }

    private synchronized void received_A(int senderID, HashMap<Integer, Float> sentA) {
        receivedA.put(senderID, sentA);
    }

    private synchronized void received_B(int senderID, HashMap<Integer, Float> sentB) {
        receivedB.put(senderID, sentB);
    }
}
