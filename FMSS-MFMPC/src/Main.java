import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        // 1. read configurations
        ReadConfigurations.readConfigurations(args);

        // 2. read training data and test data
        ReadData.readData();

        // 3. apply initialization
        Initialization.initialization();

        // 4. start server
        Server server = new Server();

        // 5. train
        long TIME_START_TRAIN = System.currentTimeMillis();
        server.train(Data.num_iterations);
        long TIME_FINISH_TRAIN = System.currentTimeMillis();
        System.out.println("Elapsed Time (train): " + Float.toString((TIME_FINISH_TRAIN - TIME_START_TRAIN) / 1000F) + "s");

        // 6. test
        Test.test();
    }
}