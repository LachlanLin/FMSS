import java.util.Arrays;

public class ReadConfigurations {
    public static void readConfigurations(String[] args) {
        // read the configurations
        for (int k = 0; k < args.length; k++) {
            if (args[k].equals("-d")) Data.d = Integer.parseInt(args[++k]);
            else if (args[k].equals("-alpha")) Data.alpha = Float.parseFloat(args[++k]);
            else if (args[k].equals("-lambda")) Data.lambda = Float.parseFloat(args[++k]);
            else if (args[k].equals("-gamma")) Data.gamma = Float.parseFloat(args[++k]);
            else if (args[k].equals("-xi")) Data.xi = Float.parseFloat(args[++k]);
            else if (args[k].equals("-L")) Data.L = Integer.parseInt(args[++k]);
            else if (args[k].equals("-rho")) Data.rho = Integer.parseInt(args[++k]);
            else if (args[k].equals("-c")) Data.c = Integer.parseInt(args[++k]);
            else if (args[k].equals("-fnTrainData")) Data.fnTrainData = args[++k];
            else if (args[k].equals("-fnTestData")) Data.fnTestData = args[++k];
            else if (args[k].equals("-n")) Data.n = Integer.parseInt(args[++k]);
            else if (args[k].equals("-m")) Data.m = Integer.parseInt(args[++k]);
            else if (args[k].equals("-num_iterations")) Data.num_iterations = Integer.parseInt(args[++k]);
            else if (args[k].equals("-topK")) Data.topK = Integer.parseInt(args[++k]);
        }

        // === Print the configurations
        System.out.println(Arrays.toString(args));
        System.out.println("d: " + Data.d);
        System.out.println("alpha: " + Data.alpha);
        System.out.println("lambda: " + Data.lambda);
        System.out.println("gamma: " + Data.gamma);
        System.out.println("xi: " + Data.xi);
        System.out.println("L: " + Data.L);
        System.out.println("rho: " + Data.rho);
        System.out.println("c: " + Data.c);
        System.out.println("fnTrainData: " + Data.fnTrainData);
        System.out.println("fnTestData: " + Data.fnTestData);
        System.out.println("n: " + Data.n);
        System.out.println("m: " + Data.m);
        System.out.println("num_iterations: " + Data.num_iterations);
        System.out.println("topK: " + Data.topK);
    }
}
