import java.util.LinkedList;
import java.util.List;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.HashSet;

public class Client implements Runnable {
    private int No; // No of Client u
    private int iter; // iteration number

    private List<Integer> I_u_sample; // sample item set of client u
    private List<Integer> I_u; // rated item set of client u
    private HashSet<Integer> clientsToSent;

    // private parameters
    private float[] eta_u_gradient;

    // intermediate values sent to the server
    private HashMap<Integer, float[]> VGradient;
    private HashMap<Integer, float[]> WGradient;
    private HashMap<Integer, Float> biGradient;
    private float[] etaGradient;
    private HashMap<Integer, Integer> countVAndBi;
    private HashMap<Integer, Integer> countW;

    // received intermediate values from other clients
    private HashMap<Integer, HashMap<Integer, float[]>> receivedVG;
    private HashMap<Integer, HashMap<Integer, float[]>> receivedWG;
    private HashMap<Integer, float[]> receivedEtaG;
    private HashMap<Integer, HashMap<Integer, Float>> receivedBiG;
    private HashMap<Integer, HashMap<Integer, Integer>> receivedVC;
    private HashMap<Integer, HashMap<Integer, Integer>> receivedWC;

    // initialization of client u
    Client(int userID) {
        No = userID;
        iter = 0;

        I_u = new LinkedList<Integer>(Data.TrainData.get(this.No));
        I_u_sample = new LinkedList<Integer>(Data.ItemSetWhole);
        I_u_sample.removeAll(this.I_u);
        clientsToSent = new HashSet<Integer>();

        eta_u_gradient = new float[Data.L + 1];
        VGradient = new HashMap<Integer, float[]>();
        WGradient = new HashMap<Integer, float[]>();
        biGradient = new HashMap<Integer, Float>();
        etaGradient = new float[Data.L + 1];
        countVAndBi = new HashMap<Integer, Integer>();
        countW = new HashMap<Integer, Integer>();

        receivedVG = new HashMap<Integer, HashMap<Integer, float[]>>();
        receivedWG = new HashMap<Integer, HashMap<Integer, float[]>>();
        receivedEtaG = new HashMap<Integer, float[]>();
        receivedBiG = new HashMap<Integer, HashMap<Integer, Float>>();
        receivedVC = new HashMap<Integer, HashMap<Integer, Integer>>();
        receivedWC = new HashMap<Integer, HashMap<Integer, Integer>>();
    }

    @Override
    public void run() {
        if (Data.step == 0) {
            run0();
        } else {
            run1();
        }
    }

    private void run0() {
        // clear the intermediate value of the previous iteration
        this.VGradient.clear();
        this.WGradient.clear();
        this.biGradient.clear();
        for (int l = 1; l < Data.L + 1; ++l) {
            eta_u_gradient[l] = 0f;
            etaGradient[l] = 0f;
        }
        this.countVAndBi.clear();
        this.countW.clear();

        this.clientsToSent.clear();
        // calculate gradient via rated items
        HashSet<Integer> fakeItemSetV = new HashSet<Integer>();
        for (int i : this.I_u) {
            if (this.I_u.indexOf(i) < Data.L) {
                continue;
            }
            int ItemSetSize = this.I_u.size();
            ArrayList<Integer> pre = new ArrayList<Integer>();
            pre.add(0);
            // pre:save the short term markov_chain items [t-1,t-2,t-3...]
            for (int r = 1; r <= Data.L; ++r) {
                int temp = this.I_u.get(this.I_u.indexOf(i) - r);
                pre.add(temp);
            }

            // got the negative sample:j
            int j = -1;
            while (true) {
                j = (int) (Math.floor(Math.random() * Data.m) + 1); // [1,m]
                if (Data.ItemSetWhole.contains(j) && (!this.I_u.contains(j))) {
                    break;
                }
            }
            fakeItemSetV.add(j);
            // ----- normalization
            float normalizationFactor_plus = (float) Math.pow(ItemSetSize - 1f + 0.0001f, Data.alpha);
            float normalizationFactor_plus_negative = (float) Math.pow(ItemSetSize - 0f + 0.0001f, Data.alpha);

            float[] U_u_i = new float[Data.d];
            float[] U_u_i_negative = new float[Data.d];

            // long term preferences
            for (int i2 : this.I_u) {
                if (i2 != i) {
                    for (int f = 0; f < Data.d; f++) {
                        U_u_i[f] += Data.W[i2][f];
                    }
                }
            }

            for (int i2 : this.I_u) {
                for (int f = 0; f < Data.d; f++) {
                    U_u_i_negative[f] += Data.W[i2][f];
                }
            }
            for (int f = 0; f < Data.d; f++) {
                U_u_i[f] = U_u_i[f] / normalizationFactor_plus;
                U_u_i_negative[f] = U_u_i_negative[f] / normalizationFactor_plus_negative;
            }

            // short term dynamics
            for (int f = 0; f < Data.d; f++) {
                float temp = 0f;
                for (int l = 1; l <= Data.L; l++) {
                    temp += Data.W[pre.get(l)][f] * (Data.eta_u[this.No][l] + Data.eta[l]);
                }

                U_u_i[f] += temp;
                U_u_i_negative[f] += temp;
            }

            // ----- $r_{ui}$
            float r_ui = Data.bi[i];
            float r_ui_negative = Data.bi[j];

            for (int f = 0; f < Data.d; f++) {
                r_ui += U_u_i[f] * Data.V[i][f];
                r_ui_negative += U_u_i_negative[f] * Data.V[j][f];
            }

            // ----- $e_{ui}$
            float e_uij = (r_ui - r_ui_negative);
            e_uij = 1f / (1 + (float) Math.pow(Math.E, e_uij));

            if (biGradient.containsKey(i)) {
                biGradient.put(i, biGradient.get(i) - e_uij + Data.lambda * Data.bi[i]);
            } else {
                biGradient.put(i, -e_uij + Data.lambda * Data.bi[i]);
            }

            if (!countVAndBi.containsKey(i)) {
                this.countVAndBi.put(i, 1);
            }

            if (biGradient.containsKey(j)) {
                biGradient.put(j, biGradient.get(j) + e_uij + Data.lambda * Data.bi[j]);
            } else {
                biGradient.put(j, e_uij + Data.lambda * Data.bi[j]);
            }
            if (!countVAndBi.containsKey(j)) {
                this.countVAndBi.put(j, 1);
            }

            float[] tmp_gradient_v = new float[Data.d];
            for (int f = 0; f < Data.d; f++) {
                tmp_gradient_v[f] = -e_uij * U_u_i[f] + Data.lambda * Data.V[i][f];
            }
            if (VGradient.containsKey(i)) {
                for (int f = 0; f < Data.d; f++) {
                    VGradient.get(i)[f] += tmp_gradient_v[f];
                }
            } else {
                VGradient.put(i, tmp_gradient_v);
            }

            float[] tmp_gradient_v_negative = new float[Data.d];
            for (int f = 0; f < Data.d; f++) {
                tmp_gradient_v_negative[f] = e_uij * U_u_i_negative[f] + Data.lambda * Data.V[j][f];
            }
            if (VGradient.containsKey(j)) {
                for (int f = 0; f < Data.d; f++) {
                    VGradient.get(j)[f] += tmp_gradient_v_negative[f];
                }
            } else {
                VGradient.put(j, tmp_gradient_v_negative);
            }

            for (int l = 1; l <= Data.L; l++) {
                float term = 0.0f;
                for (int f = 0; f < Data.d; f++) { // dot product
                    term += Data.W[pre.get(l)][f] * (Data.V[i][f] - Data.V[j][f]);
                }

                etaGradient[l] += -e_uij * term + Data.lambda * Data.eta[l];
                eta_u_gradient[l] += -e_uij * term + Data.lambda * Data.eta_u[this.No][l];
            }

            float[] tmp_gradient_w = new float[Data.d];
            for (int f = 0; f < Data.d; ++f) {
                tmp_gradient_w[f] = e_uij * Data.V[j][f] / normalizationFactor_plus_negative
                        + Data.lambda * Data.W[i][f];
            }
            if (WGradient.containsKey(i)) {
                for (int f = 0; f < Data.d; f++) {
                    WGradient.get(i)[f] += tmp_gradient_w[f];
                }
            } else {
                WGradient.put(i, tmp_gradient_w);
            }

            if (iter == 0) {
                if (!countW.containsKey(i)) {
                    countW.put(i, 1);
                }
            }

            for (int i2 : this.I_u) {
                if (i2 != i && !pre.contains(i2)) {
                    float[] tmp_gradient_w2 = new float[Data.d];
                    for (int f = 0; f < Data.d; f++) {
                        tmp_gradient_w2[f] = -e_uij
                                * (Data.V[i][f] / normalizationFactor_plus
                                - Data.V[j][f] / normalizationFactor_plus_negative)
                                + Data.lambda * Data.W[i2][f];
                    }
                    if (WGradient.containsKey(i2)) {
                        for (int f = 0; f < Data.d; f++) {
                            WGradient.get(i2)[f] += tmp_gradient_w2[f];
                        }
                    } else {
                        WGradient.put(i2, tmp_gradient_w2);
                    }

                    if (iter == 0) {
                        if (!countW.containsKey(i2)) {
                            countW.put(i2, 1);
                        }
                    }
                }
            }

            // ----- update $W_{i'\cdot}$ --- short term items
            for (int l = 1; l <= Data.L; l++) {
                int item = pre.get(l);
                float[] tmp_gradient_w3 = new float[Data.d];
                for (int f = 0; f < Data.d; ++f) {
                    float temp = (Data.V[i][f] * (1f / normalizationFactor_plus + Data.eta[l] + Data.eta_u[this.No][l]) - Data.V[j][f] * (1f / normalizationFactor_plus_negative + Data.eta[l] + Data.eta_u[this.No][l]));
                    tmp_gradient_w3[f] = -e_uij * temp + Data.lambda * Data.W[item][f];
                }
                if (WGradient.containsKey(item)) {
                    for (int f = 0; f < Data.d; f++) {
                        WGradient.get(item)[f] += tmp_gradient_w3[f];
                    }
                } else {
                    WGradient.put(item, tmp_gradient_w3);
                }
                if (iter == 0) {
                    if (!countW.containsKey(item)) {
                        countW.put(item, 1);
                    }
                }
            }
        }
        // sample fake items for W with a sampling parameter of rho
        int sample_number = Math.min(Data.rho * this.I_u.size(), I_u_sample.size());
        HashSet<Integer> fakeItemSetW = new HashSet<Integer>();
        while (fakeItemSetW.size() < sample_number) {
            int randomIndex = (int) (Math.random() * this.I_u_sample.size());
            int randomItem = this.I_u_sample.get(randomIndex);
            if (Data.ItemSetWhole.contains(randomItem) && !fakeItemSetW.contains(randomItem)) {
                fakeItemSetW.add(randomItem);
                float[] fakeW = new float[Data.d];
                WGradient.put(randomItem, fakeW);
                if (iter == 0) {
                    countW.put(randomItem, 0);
                }
            }
        }

        // sample fake items for V and bi with a sampling parameter of rho - a
        while (fakeItemSetV.size() < sample_number) {
            int randomIndex = (int) (Math.random() * this.I_u_sample.size());
            int randomItem = this.I_u_sample.get(randomIndex);
            if (Data.ItemSetWhole.contains(randomItem) && !fakeItemSetV.contains(randomItem)) {
                fakeItemSetV.add(randomItem);
                float[] fakeV = new float[Data.d];
                VGradient.put(randomItem, fakeV);
                biGradient.put(randomItem, 0f);
                if (iter == 0) {
                    countVAndBi.put(randomItem, 0);
                }
            }
        }

        // sample some clients to sent random numbers, i.e., \mathcal{N}_u
        while (this.clientsToSent.size() < Data.c) {
            Integer userID = (int) (Data.n * Math.random()) + 1;
            if (Data.TrainData.containsKey(userID))
                this.clientsToSent.add(userID);
        }

        for (Integer userID : this.clientsToSent) {
            // generate random numbers of countVAndBi and send them to others.
            HashMap<Integer, Integer> sentCountVAndBi = new HashMap<Integer, Integer>();
            for (Integer itemID : this.countVAndBi.keySet()) {
                int randomNum = (int) (Math.random() * 3);
                sentCountVAndBi.put(itemID, randomNum);
                this.countVAndBi.put(itemID, this.countVAndBi.get(itemID) - randomNum);
            }
            Data.client[userID].receive_VC(this.No, sentCountVAndBi);

            if (iter == 0) {
                HashMap<Integer, Integer> sentCountW = new HashMap<Integer, Integer>();
                for (Integer itemID : this.countW.keySet()) {
                    int randomNum = (int) (Math.random() * 3);
                    sentCountW.put(itemID, randomNum);
                    this.countW.put(itemID, this.countW.get(itemID) - randomNum);
                }
                Data.client[userID].receive_WC(this.No, sentCountW);
            }

            // generate random numbers of V gradients and send them to others.
            HashMap<Integer, float[]> VSentGradient = new HashMap<Integer, float[]>();
            for (Integer itemID : this.VGradient.keySet()) {
                float[] tmp_gradient = new float[Data.d];
                for (int f = 0; f < Data.d; f++) {
                    float randomGradientV = (float) Math.random() * 3.0f;
                    tmp_gradient[f] = randomGradientV;
                    this.VGradient.get(itemID)[f] -= randomGradientV;
                }
                VSentGradient.put(itemID, tmp_gradient);
            }
            Data.client[userID].receive_VG(this.No, VSentGradient);

            // generate random numbers of W gradients and send them to others.
            HashMap<Integer, float[]> WSentGradient = new HashMap<Integer, float[]>();
            for (Integer itemID : this.WGradient.keySet()) {
                float[] tmp_gradient_w = new float[Data.d];
                for (int f = 0; f < Data.d; f++) {
                    float randomGradientW = (float) Math.random() * 3.0f;
                    tmp_gradient_w[f] = randomGradientW;
                    this.WGradient.get(itemID)[f] -= randomGradientW;
                }
                WSentGradient.put(itemID, tmp_gradient_w);
            }
            Data.client[userID].receive_WG(this.No, WSentGradient);

            // generate random numbers of bi gradients and send them to others.
            HashMap<Integer, Float> biSentGradient = new HashMap<Integer, Float>();
            for (Integer itemID : this.VGradient.keySet()) {
                float randomGradientBi = (float) Math.random() * 3.0f;
                this.biGradient.put(itemID, this.biGradient.get(itemID) - randomGradientBi);
                biSentGradient.put(itemID, randomGradientBi);
            }
            Data.client[userID].receive_BiG(this.No, biSentGradient);

            // generate random numbers of eta gradients and send them to others.
            float[] etaSentGradient = new float[Data.L + 1];
            for (int l = 1; l < Data.L + 1; l++) {
                float randomNum = (float) Math.random() * 3.0f;
                etaSentGradient[l] = randomNum;
                this.etaGradient[l] -= randomNum;
            }
            Data.client[userID].receive_etaG(this.No, etaSentGradient);
        }
    }

    private void run1() {
        // receive the countI, calculate the sum of them, and send them to the server
        for (Integer userID : receivedVC.keySet()) {
            for (Integer itemID : receivedVC.get(userID).keySet()) {
                int tmp_int = receivedVC.get(userID).get(itemID);
                if (this.countVAndBi.containsKey(itemID)) {
                    this.countVAndBi.put(itemID, this.countVAndBi.get(itemID) + tmp_int);
                } else {
                    this.countVAndBi.put(itemID, tmp_int);
                }
            }
        }
        Data.countOfVAndBiHashMap.put(this.No, this.countVAndBi);

        // receive the countIR, calculate the sum of them, and send them to the server
        if (iter == 0) {
            for (Integer userID : receivedWC.keySet()) {
                for (Integer itemID : receivedWC.get(userID).keySet()) {
                    int tmp_int = receivedWC.get(userID).get(itemID);
                    if (this.countW.containsKey(itemID)) {
                        this.countW.put(itemID, this.countW.get(itemID) + tmp_int);
                    } else {
                        this.countW.put(itemID, tmp_int);
                    }
                }
            }
            Data.countOfWHashMap.put(this.No, this.countW);
        }

        // receive the V gradients, calculate the sum of them, and send them to the server
        for (Integer userID : receivedVG.keySet()) {
            for (Integer itemID : receivedVG.get(userID).keySet()) {
                float[] tmp_v = receivedVG.get(userID).get(itemID);
                if (this.VGradient.containsKey(itemID)) {
                    float[] tmp_gradient = this.VGradient.get(itemID);
                    for (int f = 0; f < Data.d; f++) {
                        tmp_gradient[f] += tmp_v[f];
                    }
                } else {
                    float[] tmp_gradient = new float[Data.d];
                    for (int f = 0; f < Data.d; f++) {
                        tmp_gradient[f] = tmp_v[f];
                    }
                    this.VGradient.put(itemID, tmp_gradient);
                }
            }
        }
        Data.V_Gradient.put(this.No, this.VGradient);

        // receive the W gradients, calculate the sum of them, and send them to the server
        for (Integer userID : receivedWG.keySet()) {
            for (Integer itemID : receivedWG.get(userID).keySet()) {
                float[] tmp_w = receivedWG.get(userID).get(itemID);
                if (this.WGradient.containsKey(itemID)) {
                    float[] tmp_gradient = this.WGradient.get(itemID);
                    for (int f = 0; f < Data.d; f++) {
                        tmp_gradient[f] += tmp_w[f];
                    }
                } else {
                    float[] tmp_gradient = new float[Data.d];
                    for (int f = 0; f < Data.d; f++) {
                        tmp_gradient[f] = tmp_w[f];
                    }
                    this.WGradient.put(itemID, tmp_gradient);
                }
            }
        }
        Data.W_Gradient.put(this.No, this.WGradient);

        // receive the bi gradients, calculate the sum of them, and send them to the server
        for (Integer userID : receivedBiG.keySet()) {
            for (Integer itemID : receivedBiG.get(userID).keySet()) {
                float tmp_float = receivedBiG.get(userID).get(itemID);
                if (this.biGradient.containsKey(itemID)) {
                    this.biGradient.put(itemID, this.biGradient.get(itemID) + tmp_float);
                } else {
                    this.biGradient.put(itemID, tmp_float);
                }
            }
        }
        Data.bi_Gradient.put(this.No, this.biGradient);

        // receive the eta gradients, calculate the sum of them, and send them to the server
        for (Integer userID : receivedEtaG.keySet()) {
            float[] tmp_eta = receivedEtaG.get(userID);
            for (int l = 1; l < Data.L + 1; l++) {
                etaGradient[l] += tmp_eta[l];
            }
        }
        Data.eta_Gradient.put(this.No, this.etaGradient);

        // Update private parameters
        for (int l = 1; l < Data.L + 1; l++) {
            Data.eta_u[this.No][l] -= Data.gamma * this.eta_u_gradient[l] / (float) this.I_u.size();
            this.eta_u_gradient[l] = 0f;
        }

        // empty the follow sets for new iteration
        receivedVG.clear();
        receivedWG.clear();
        receivedEtaG.clear();
        receivedBiG.clear();
        receivedVC.clear();
        receivedWC.clear();

        this.iter++;
    }

    public synchronized void receive_VG(int senderID, HashMap<Integer, float[]> sentV) {
        receivedVG.put(senderID, sentV);
    }

    public synchronized void receive_WG(int senderID, HashMap<Integer, float[]> sentW) {
        receivedWG.put(senderID, sentW);
    }

    public synchronized void receive_etaG(int senderID, float[] sentEta) {
        receivedEtaG.put(senderID, sentEta);
    }

    public synchronized void receive_BiG(int senderID, HashMap<Integer, Float> sentBi) {
        receivedBiG.put(senderID, sentBi);
    }

    public synchronized void receive_VC(int senderID, HashMap<Integer, Integer> sentVC) {
        receivedVC.put(senderID, sentVC);
    }

    public synchronized void receive_WC(int senderID, HashMap<Integer, Integer> sentWC) {
        receivedWC.put(senderID, sentWC);
    }

}