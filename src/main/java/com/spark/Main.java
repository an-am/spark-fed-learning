package com.spark;

import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.api.java.function.ReduceFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.reflect.ClassTag$;

import java.util.*;

public class Main {

    private static final double FITTED_LAMBDA_INCOME = 0.3026418664067109;
    private static final double FITTED_LAMBDA_WEALTH =  0.1336735055366279;

    private static final int numIterations = 10;
    private static final int numEpochs = 10;

    private static final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345) //include a random seed for reproducibility
            .activation(Activation.RELU)
            .weightInit(WeightInit.ZERO)
            .updater(new Adam())
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder() //create the first input layer.
                    .nIn(9)
                    .nOut(128)
                    .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) //create the second input layer
                    .nIn(128)
                    .nOut(1)
                    .activation(Activation.RELU)
                    .build())
            .build();

    // Full schema for downstream processing (if needed)
    public static final StructType fullSchema = new StructType(new StructField[]{
            new StructField("Id", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("Age", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("Gender", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("FamilyMembers", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("FinancialEducation", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("RiskPropensity", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("Income", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("Wealth", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("IncomeInvestment", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("AccumulationInvestment", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("FinancialStatus", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("ClientId", DataTypes.IntegerType, true, Metadata.empty())
    });

    // schema: features array + label
    public static final StructType featureSchema = new StructType(new StructField[]{
            new StructField("features",
                    DataTypes.createArrayType(DataTypes.FloatType,false),
                    false,
                    Metadata.empty()),
            new StructField("label",
                    DataTypes.FloatType,
                    false,
                    Metadata.empty())
    });


    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("SparkFedLearning")
                .master("spark://Tonys-MacBook-Pro.local:7077")
                .config("spark.jars.excludes", "org.slf4j:slf4j-api")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        MultiLayerNetwork globalModel = new MultiLayerNetwork(conf);
        globalModel.init();

        String CSV_PATH = "Needs.csv";
        Dataset<Row> fullDataset = spark
                .read()
                .option("header", "true")
                .schema(fullSchema)
                .csv(CSV_PATH);

        Dataset<Row> clients = fullDataset
                .repartition(10)
                .mapPartitions((MapPartitionsFunction<Row, Row>) iter -> {
                    // Data prep

                    List<Row> rows = new ArrayList<>();
                    iter.forEachRemaining(rows::add);

                    int num_features = 9;
                    int n = rows.size();

                    float[][] data = new float[n][num_features];
                    for (int i = 0; i < n; i++) {
                        Row r = rows.get(i);
                        // ---- build feature vector -----------------------------------------
                        data[i][0] = r.getInt(1);                       // Age
                        data[i][1] = r.getInt(2);                       // Gender
                        data[i][2] = r.getInt(3);                       // FamilyMembers
                        data[i][3] = r.getFloat(4);                     // FinancialEducation
                        float income = r.getFloat(6);
                        float wealth = r.getFloat(7);
                        data[i][4] = (float) ((Math.pow(income,  FITTED_LAMBDA_INCOME) - 1.0)
                                / FITTED_LAMBDA_INCOME);
                        data[i][5] = (float) ((Math.pow(wealth,  FITTED_LAMBDA_WEALTH) - 1.0)
                                / FITTED_LAMBDA_WEALTH);
                        data[i][6] = r.getInt(8);                       // IncomeInvestment
                        data[i][7] = r.getInt(9);                       // AccumulationInvestment
                        data[i][8] = r.getFloat(4) * (float) Math.log(wealth); // FinancialStatus
                    }

                    INDArray dataMatrix = Nd4j.create(data);

                    // Scaler (StandardScaler behaviour)
                    INDArray meanVec = dataMatrix.mean(0);
                    INDArray stdVec  = dataMatrix.std(0);
                    INDArray scaled  = dataMatrix.subRowVector(meanVec)
                                                 .divRowVector(stdVec);

                    List<Row> out = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        float[] rowVec = scaled.getRow(i).toFloatVector();

                        float lbl = rows.get(i).getFloat(5);   // RiskPropensity column

                        out.add(RowFactory.create(rowVec, lbl));
                    }
                    return out.iterator();

                }, Encoders.row(featureSchema) )
                .persist(StorageLevel.MEMORY_AND_DISK());

        // FedAvg algorithm
        for (int i = 0; i < numIterations; i++) {

            // Broadcast the global model's weights
            Broadcast<INDArray> globalWeights = spark
                    .sparkContext()
                    .broadcast(globalModel.params(), ClassTag$.MODULE$.apply(INDArray.class));

            // Local train
            Dataset<INDArray> localWeights = clients
                    .mapPartitions((MapPartitionsFunction<Row, INDArray>) iter -> {

                        List<float[]> featList = new ArrayList<>(); // Features
                        List<Float>   labList  = new ArrayList<>(); // Labels

                        while (iter.hasNext()) {
                            Row r = iter.next();

                            List<Float> feats = r.getList(0);   // First column = features array
                            float[] f = new float[feats.size()];
                            for (int k = 0; k < f.length; k++) f[k] = feats.get(k);
                            featList.add(f);

                            labList.add(r.getFloat(1)); // Second column = label
                        }

                        int rows = featList.size();
                        int cols = featList.get(0).length;

                        INDArray featureMtx = Nd4j.create(rows, cols);
                        INDArray labelVec   = Nd4j.create(rows, 1);

                        for (int j = 0; j < rows; j++) {
                            featureMtx.putRow(j, Nd4j.create(featList.get(j)));
                            labelVec.putScalar(j, 0, labList.get(j));
                        }

                        DataSet d = new DataSet(featureMtx, labelVec);

                        INDArray w = globalWeights.value();
                        MultiLayerNetwork local = new MultiLayerNetwork(conf);
                        local.init(w, false);


                        for (int epoch = 0; epoch < numEpochs; epoch++) {
                            IteratorDataSetIterator it = new IteratorDataSetIterator(d.iterator(), 64);
                            local.fit(it);  // local training
                        }

                        IteratorDataSetIterator it = new IteratorDataSetIterator(d.iterator(), 64);
                        RegressionEvaluation eval = local.evaluateRegression(it);

                        System.out.println(eval.stats());

                        w = local.params();

                        return Collections.singletonList(w).iterator();
                    }, Encoders.kryo(INDArray.class));


                // ---  Driver‑side FedAvg  ---------------------------------
                // 1. Sum all weight maps
                INDArray sum = localWeights.reduce(
                        (ReduceFunction<INDArray>) ( a,  b) -> {
                            b.addi(a);
                            return b;
                        });

                // 2. Count how many local models contributed
                long numModels = localWeights.count();

                // 3. Divide by N  (sum = Σ w_i  →  mean = Σ w_i / N)
                sum.divi(numModels);

                // 4. Update the global model with the averaged parameters
                globalModel.setParams(sum);

            System.out.println("Iteration " + i + " completed.");

        }
    }
}
