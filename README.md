# Federated Learning with Apache Spark in Java

In this implementation, a **fixed set** of federated clients train their local neural network on their own dataset, satisfying **data locality** principles.
A **server** aggregates new model weights, following the **Federated Averaging** algorithm, for a total of 10 rounds.
