import java.util.LinkedList;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;

public class Client implements Runnable {
    private int No; // No of Client u
    private int iter; // iteration number

    private List<Integer> I_u_sample; // sample item set of client u
    private List<Integer> I_u; // rated item set of client u
    private HashSet<Integer> clientsToSent;

    // private parameters
    private float[] UGradient;// user gradient
    private float b_u_gradient; // user bias gradient;

    // intermediate values sent to the server
    private HashMap<Integer, float[]> VGradient;
    private HashMap<Integer, HashMap<Integer, float[]>> MGradient;
    private HashMap<Integer, Float> biGradient;
    private float muGradient;
    private HashMap<Integer, Integer> countI;
    private HashMap<Integer, HashMap<Integer, Integer>> countIR;

    // received intermediate values from other clients
    public HashMap<Integer, HashMap<Integer, float[]>> receivedVG;
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, float[]>>> receivedMG;
    public HashMap<Integer, HashMap<Integer, Float>> receivedBiG;
    public HashMap<Integer, Float> receivedMuG;
    public HashMap<Integer, HashMap<Integer, Integer>> receivedVC;
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>> receivedVRC;

    // initialization of client u
    Client(int userID) {
        No = userID;
        iter = 0;

        I_u = new LinkedList<Integer>(Data.I_u.get(this.No));
        I_u_sample = new LinkedList<Integer>(Data.I);
        I_u_sample.removeAll(this.I_u);
        clientsToSent = new HashSet<Integer>();

        UGradient = new float[Data.d];
        b_u_gradient = 0f;
        VGradient = new HashMap<Integer, float[]>();
        MGradient = new HashMap<Integer, HashMap<Integer, float[]>>();
        biGradient = new HashMap<Integer, Float>();
        muGradient = 0f;
        countI = new HashMap<Integer, Integer>();
        countIR = new HashMap<Integer, HashMap<Integer, Integer>>();

        receivedVG = new HashMap<Integer, HashMap<Integer, float[]>>();
        receivedMG = new HashMap<Integer, HashMap<Integer, HashMap<Integer, float[]>>>();
        receivedBiG = new HashMap<Integer, HashMap<Integer, Float>>();
        receivedMuG = new HashMap<Integer, Float>();
        receivedVC = new HashMap<Integer, HashMap<Integer, Integer>>();
        receivedVRC = new HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>>();
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
        this.MGradient.clear();
        this.biGradient.clear();
        this.muGradient = 0;
        this.countI.clear();
        this.countIR.clear();

        this.clientsToSent.clear();
        // calculate gradient via rated items
        for (int i : this.I_u) {
            if (iter == 0) {
                countI.put(i, 1);
                int ratingID = (int) (float) Data.trainingDataMap.get(this.No).get(i);
                HashMap<Integer, Integer> tmp = new HashMap<Integer, Integer>();
                tmp.put(ratingID, 1);
                countIR.put(i, tmp);
            }

            float[] tilde_Uu_g = new float[Data.d];
            float[] tilde_Uu = new float[Data.d];

            for (int g = 1; g <= Data.p; g++) {
                int ratingSetSize = Data.I_r_u.get(this.No).get(g).size();
                if (ratingSetSize > 0) {
                    HashSet<Integer> itemSet = Data.I_r_u.get(this.No).get(g);
                    float explicit_feedback_num_u_sqrt = 0;
                    if (itemSet.contains(i)) {
                        if (itemSet.size() > 1) {
                            explicit_feedback_num_u_sqrt = (float) Math.sqrt(ratingSetSize - 1);
                        }
                    } else {
                        explicit_feedback_num_u_sqrt = (float) Math.sqrt(ratingSetSize);
                    }

                    if (explicit_feedback_num_u_sqrt > 0) {
                        // --- aggregation
                        for (int i2 : itemSet) {
                            if (i2 != i) {
                                for (int f = 0; f < Data.d; f++) {
                                    tilde_Uu_g[f] += Data.M[i2][g][f];
                                }
                            }
                        }

                        // --- normalization
                        for (int f = 0; f < Data.d; f++) {
                            tilde_Uu_g[f] = tilde_Uu_g[f] / explicit_feedback_num_u_sqrt;
                            tilde_Uu[f] += tilde_Uu_g[f];
                            tilde_Uu_g[f] = 0;
                        }
                    }
                }
            }

            // prediction and error
            float pred = 0;
            float error = 0;
            for (int f = 0; f < Data.d; f++) {
                pred += Data.U[this.No][f] * Data.V[i][f] + tilde_Uu[f] * Data.V[i][f];
            }
            pred += Data.mu + Data.b_u[this.No] + Data.b_i[i];
            error = Data.trainingDataMap.get(this.No).get(i) - pred;

            this.muGradient += -error;

            this.b_u_gradient += -error + Data.lambda * Data.b_u[this.No];
            biGradient.put(i, -error + Data.lambda * Data.b_i[i]);

            // the gradients of private parameters
            for (int f = 0; f < Data.d; f++) {
                this.UGradient[f] += -error * Data.V[i][f] + Data.lambda * Data.U[this.No][f];
            }

            // the gradients of shared parameters
            float[] tmp_v_gradient = new float[Data.d];
            for (int f = 0; f < Data.d; f++) {
                tmp_v_gradient[f] = -error * (Data.U[this.No][f] + tilde_Uu[f]) + Data.lambda * Data.V[i][f];
            }
            VGradient.put(i, tmp_v_gradient);

            for (int g = 1; g <= Data.p; g++) {
                int ratingSetSize = Data.I_r_u.get(this.No).get(g).size();
                if (ratingSetSize > 0) {
                    HashSet<Integer> itemSet = Data.I_r_u.get(this.No).get(g);
                    float explicit_feedback_num_u_sqrt = 0;
                    if (itemSet.contains(i)) {
                        if (itemSet.size() > 1) {
                            explicit_feedback_num_u_sqrt = (float) Math.sqrt(ratingSetSize - 1);
                        }
                    } else {
                        explicit_feedback_num_u_sqrt = (float) Math.sqrt(ratingSetSize);
                    }

                    if (explicit_feedback_num_u_sqrt > 0) {
                        for (int i_ : itemSet) {
                            if (i_ != i) {
                                float[] tmp_m_gradient = new float[Data.d];
                                for (int f = 0; f < Data.d; f++) {
                                    tmp_m_gradient[f] = (-error * Data.V[i][f] / explicit_feedback_num_u_sqrt
                                            + Data.lambda * Data.M[i_][g][f]);
                                }
                                if (MGradient.containsKey(i_)) {
                                    if (MGradient.get(i_).containsKey(g)) {
                                        float[] tmp_m_gradient_2 = new float[Data.d];
                                        for (int f = 0; f < Data.d; f++) {
                                            tmp_m_gradient_2[f] = MGradient.get(i_).get(g)[f] + tmp_m_gradient[f];
                                        }
                                        MGradient.get(i_).put(g, tmp_m_gradient_2);
                                    } else {
                                        MGradient.get(i_).put(g, tmp_m_gradient);
                                    }
                                } else {
                                    HashMap<Integer, float[]> tmp = new HashMap<Integer, float[]>();
                                    tmp.put(g, tmp_m_gradient);
                                    MGradient.put(i_, tmp);
                                }
                            }
                        }
                    }
                }
            }
        }
        // sample fake ratings for rated items
        for (int i : this.I_u) {
            int ratingID = (int) (float) Data.trainingDataMap.get(this.No).get(i);
            // sample fake ratings
            int fake_rating_num = Math.min(Data.rho, Data.p - 1);
            HashSet<Integer> fakeRatingSet = new HashSet<Integer>();
            while (fakeRatingSet.size() < fake_rating_num) {
                int random_ratingID = (int) (Data.p * Math.random()) + 1;
                if (random_ratingID != ratingID) {
                    fakeRatingSet.add(random_ratingID);
                }
            }
            for (Integer r : fakeRatingSet) {
                float[] tmp_m_gradient = new float[Data.d];
                MGradient.get(i).put(r, tmp_m_gradient);
                if (iter == 0) {
                    countIR.get(i).put(r, 0);
                }
            }
        }

        // sample fake items and fake ratings
        int sample_number = Math.min(Data.rho * this.I_u.size(), I_u_sample.size());
        HashSet<Integer> fakeItemSet = new HashSet<Integer>();
        while (fakeItemSet.size() < sample_number) {
            int randomIndex = (int) (Math.random() * this.I_u_sample.size());
            fakeItemSet.add(this.I_u_sample.get(randomIndex));
        }
        for (Integer i : fakeItemSet) {
            float[] fakeLatentDimensionsList = new float[Data.d];
            VGradient.put(i, fakeLatentDimensionsList);
            biGradient.put(i, 0f);
            if (iter == 0) {
                countI.put(i, 0);
            }

            // sample fake ratings for fake items
            int fake_rating_num = Math.min(Data.rho + 1, Data.p);
            HashSet<Integer> fakeRatingSet = new HashSet<Integer>();
            while (fakeRatingSet.size() < fake_rating_num) {
                int random_ratingID = (int) (Data.p * Math.random()) + 1;
                fakeRatingSet.add(random_ratingID);
            }
            for (Integer r : fakeRatingSet) {
                float[] tmp_m_gradient = new float[Data.d];
                if (MGradient.containsKey(i)) {
                    MGradient.get(i).put(r, tmp_m_gradient);
                    if (iter == 0) {
                        countIR.get(i).put(r, 0);
                    }

                } else {
                    HashMap<Integer, float[]> tmp = new HashMap<Integer, float[]>();
                    tmp.put(r, tmp_m_gradient);
                    MGradient.put(i, tmp);
                    if (iter == 0) {
                        HashMap<Integer, Integer> tmp_2 = new HashMap<Integer, Integer>();
                        tmp_2.put(r, 0);
                        countIR.put(i, tmp_2);
                    }
                }

            }
        }

        // sample some clients to sent random numbers, i.e., \mathcal{N}_u
        while (this.clientsToSent.size() < Data.c) {
            Integer userID = (int) (Data.n * Math.random()) + 1;
            this.clientsToSent.add(userID);
        }

        // generate random numbers and send them to other clients
        for (Integer userID : this.clientsToSent) {
            if (iter == 0) {
                // generate random numbers of countI and send them to others.
                HashMap<Integer, Integer> sentCountI = new HashMap<Integer, Integer>();
                for (Integer itemID : this.countI.keySet()) {
                    int randomCountI = (int) (Math.random() * 3);
                    sentCountI.put(itemID, randomCountI);
                    this.countI.put(itemID, this.countI.get(itemID) - randomCountI);
                }
                Data.client[userID].receive_VC(this.No, sentCountI);
            }

            // generate random numbers of countIR and send them to others.
            if (iter == 0) {
                HashMap<Integer, HashMap<Integer, Integer>> sentCountIR = new HashMap<Integer, HashMap<Integer, Integer>>();
                for (Integer itemID : this.countIR.keySet()) {
                    for (Integer ratingID : this.countIR.get(itemID).keySet()) {
                        int randomCountIR = (int) (Math.random() * 3);
                        if (sentCountIR.containsKey(itemID)) {
                            sentCountIR.get(itemID).put(ratingID, randomCountIR);
                        } else {
                            HashMap<Integer, Integer> tmp = new HashMap<Integer, Integer>();
                            tmp.put(ratingID, randomCountIR);
                            sentCountIR.put(itemID, tmp);
                        }
                        this.countIR.get(itemID).put(ratingID, this.countIR.get(itemID).get(ratingID) - randomCountIR);
                    }
                }
                Data.client[userID].receive_VRC(this.No, sentCountIR);
            }

            // generate random numbers of V gradients and send them to others.
            HashMap<Integer, float[]> VSentGradient = new HashMap<Integer, float[]>();
            for (Integer itemID : this.VGradient.keySet()) {
                float[] tmp_gradient_v = new float[Data.d];
                for (int f = 0; f < Data.d; f++) {
                    float randomGradientV = (float) Math.random() * 3.0f;
                    tmp_gradient_v[f] = randomGradientV;
                    this.VGradient.get(itemID)[f] -= randomGradientV;
                }
                VSentGradient.put(itemID, tmp_gradient_v);
            }
            Data.client[userID].receive_VG(this.No, VSentGradient);

            // generate random numbers of M gradients and send them to others.
            HashMap<Integer, HashMap<Integer, float[]>> MSentGradient = new HashMap<Integer, HashMap<Integer, float[]>>();
            for (Integer itemID : this.MGradient.keySet()) {
                for (Integer ratingID : this.MGradient.get(itemID).keySet()) {
                    float[] tmp_gradient_m = new float[Data.d];
                    for (int f = 0; f < Data.d; f++) {
                        float randomGradientM = (float) Math.random() * 3.0f;
                        tmp_gradient_m[f] = randomGradientM;
                        this.MGradient.get(itemID).get(ratingID)[f] -= randomGradientM;
                    }
                    if (MSentGradient.containsKey(itemID)) {
                        MSentGradient.get(itemID).put(ratingID, tmp_gradient_m);
                    } else {
                        HashMap<Integer, float[]> tmp = new HashMap<Integer, float[]>();
                        tmp.put(ratingID, tmp_gradient_m);
                        MSentGradient.put(itemID, tmp);
                    }
                }
            }
            Data.client[userID].receive_MG(this.No, MSentGradient);

            // generate random numbers of b_i gradients and send them to others.
            HashMap<Integer, Float> biSentGradient = new HashMap<Integer, Float>();
            for (Integer itemID : this.VGradient.keySet()) {
                float tmp_gradient_b_i = 0;
                float randomGradientBi = (float) Math.random() * 3.0f;
                tmp_gradient_b_i = randomGradientBi;
                this.biGradient.put(itemID, this.biGradient.get(itemID) - randomGradientBi);
                biSentGradient.put(itemID, tmp_gradient_b_i);
            }
            Data.client[userID].receive_BiG(this.No, biSentGradient);

            // generate random numbers of mu gradients and send them to others.
            float randomGradientMu = (float) Math.random() * 3.0f;
            this.muGradient -= randomGradientMu;
            Data.client[userID].receive_MuG(this.No, randomGradientMu);
        }
    }

    private void run1() {
        // receive the countI, calculate the sum of them, and send them to the server
        if (iter == 0) {
            for (Integer userID : receivedVC.keySet()) {
                for (Integer itemID : receivedVC.get(userID).keySet()) {
                    int tmp_int = receivedVC.get(userID).get(itemID);
                    if (this.countI.containsKey(itemID)) {
                        this.countI.put(itemID, this.countI.get(itemID) + tmp_int);
                    } else {
                        this.countI.put(itemID, tmp_int);
                    }
                }
            }
            Data.countIHashMap.put(this.No, this.countI);
        }

        // receive the countIR, calculate the sum of them, and send them to the server
        if (iter == 0) {
            for (Integer userID : receivedVRC.keySet()) {
                for (Integer itemID : receivedVRC.get(userID).keySet()) {
                    for (Integer ratingID : receivedVRC.get(userID).get(itemID).keySet()) {
                        int tmp_int = receivedVRC.get(userID).get(itemID).get(ratingID);
                        if (countIR.containsKey(itemID)) {
                            HashMap<Integer, Integer> tmp = countIR.get(itemID);
                            if (tmp.containsKey(ratingID)) {
                                tmp.put(ratingID, tmp.get(ratingID) + tmp_int);
                            } else {
                                tmp.put(ratingID, tmp_int);
                            }
                        } else {
                            HashMap<Integer, Integer> tmp = new HashMap<Integer, Integer>();
                            tmp.put(ratingID, tmp_int);
                            countIR.put(itemID, tmp);
                        }
                    }
                }
            }
            Data.countIRHashMap.put(this.No, this.countIR);
        }

        // receive the V gradients, calculate the sum of them, and send them to the server
        for (Integer userID : receivedVG.keySet()) {
            for (Integer itemID : receivedVG.get(userID).keySet()) {
                float[] tmp_v = receivedVG.get(userID).get(itemID);
                if (this.VGradient.containsKey(itemID)) {
                    float[] tmp_gradient_v = this.VGradient.get(itemID);
                    for (int f = 0; f < Data.d; f++) {
                        tmp_gradient_v[f] += tmp_v[f];
                    }
                } else {
                    float[] tmp_gradient_v = new float[Data.d];
                    for (int f = 0; f < Data.d; f++) {
                        tmp_gradient_v[f] = tmp_v[f];
                    }
                    this.VGradient.put(itemID, tmp_gradient_v);
                }
            }
        }
        Data.V_Gradient.put(this.No, this.VGradient);

        // receive the M gradients, calculate the sum of them, and send them to the server
        for (Integer userID : receivedMG.keySet()) {
            for (Integer itemID : receivedMG.get(userID).keySet()) {
                for (Integer ratingID : receivedMG.get(userID).get(itemID).keySet()) {
                    float[] tmp_m = receivedMG.get(userID).get(itemID).get(ratingID);
                    if (this.MGradient.containsKey(itemID)) {
                        if (this.MGradient.get(itemID).containsKey(ratingID)) {
                            float[] tmp_gradient = this.MGradient.get(itemID).get(ratingID);
                            for (int f = 0; f < Data.d; f++) {
                                tmp_gradient[f] += tmp_m[f];
                            }
                        } else {
                            float[] tmp_gradient = new float[Data.d];
                            for (int f = 0; f < Data.d; f++) {
                                tmp_gradient[f] = tmp_m[f];
                            }
                            this.MGradient.get(itemID).put(ratingID, tmp_gradient);
                        }
                    } else {
                        float[] tmp_gradient = new float[Data.d];
                        for (int f = 0; f < Data.d; f++) {
                            tmp_gradient[f] = tmp_m[f];
                        }
                        HashMap<Integer, float[]> tmp = new HashMap<Integer, float[]>();
                        tmp.put(ratingID, tmp_gradient);
                        this.MGradient.put(itemID, tmp);
                    }
                }
            }
        }
        Data.M_Gradient.put(this.No, this.MGradient);

        // receive the b_i, calculate the sum of them, and send them to the server
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
        Data.b_i_Gradient.put(this.No, this.biGradient);

        // receive the mu, calculate the sum of them, and send them to the server
        for (Integer userID : receivedMuG.keySet()) {
            this.muGradient += receivedMuG.get(userID);
        }
        Data.mu_Gradient.put(this.No, this.muGradient);

        // Update private parameters
        for (int f = 0; f < Data.d; f++) {
            Data.U[this.No][f] -= Data.gamma * this.UGradient[f] / (float) this.I_u.size();
            this.UGradient[f] = 0f;
        }
        Data.b_u[this.No] -= Data.gamma * this.b_u_gradient / (float) this.I_u.size();
        this.b_u_gradient = 0f;

        // empty the follow sets for new iteration
        receivedVG.clear();
        receivedMG.clear();
        receivedBiG.clear();
        receivedMuG.clear();
        receivedVC.clear();
        receivedVRC.clear();

        this.iter++;
    }

    public synchronized void receive_VG(int senderID, HashMap<Integer, float[]> sentV) {
        receivedVG.put(senderID, sentV);
    }

    public synchronized void receive_MG(int senderID, HashMap<Integer, HashMap<Integer, float[]>> sentM) {
        receivedMG.put(senderID, sentM);
    }

    public synchronized void receive_BiG(int senderID, HashMap<Integer, Float> sentBi) {
        receivedBiG.put(senderID, sentBi);
    }

    public synchronized void receive_MuG(int senderID, float sentMu) {
        receivedMuG.put(senderID, sentMu);
    }

    public synchronized void receive_VC(int senderID, HashMap<Integer, Integer> sentVC) {
        receivedVC.put(senderID, sentVC);
    }

    public synchronized void receive_VRC(int senderID, HashMap<Integer, HashMap<Integer, Integer>> sentVRC) {
        receivedVRC.put(senderID, sentVRC);
    }
}