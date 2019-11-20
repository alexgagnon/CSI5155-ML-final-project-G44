# Reference Paper Notes

- focus on characterization of network traffic base on TCP/IP layer, and further into application-layer protocols (including popular web-based applications)
- VPNs are becoming popular remote access communication methods
- VPNs are governed by IP Sec, used to tunnel already encrypted IP traffic to guarantee secure remote access to servers
- goal is to classify network traffic which is encrypted and tunnelled through a VPN or regular encrypted, but not VPN tunnelled traffic
- They use:
  - logistic regression
  - SVM
  - Naive Bayes
  - kNN
  - Random Forest
  - Gradient Boosting Tree
  - They recommend optimized RF and GBT models are best in terms of high accuracy and low overfitting.
  - Found features that achieve 90% accuracy

Classification of IP traffic can be defined based on various
functions such as encryption of traffic (using HTTPS) during communication from source
to destination based on the IP address, protocol encapsulation using IP Security (IPSec)
protocol (or VPN tunnelling), application-specific port numbers (HTTP, FTP or SSH) or
specific applications such as Skype, Facebook, Gmail etc. Some of these real-time
applications are complex since they use multiple services, which leads to identification
of the application and the specific task associated with it.

VPn has become preferred NAM for routing traffic between two end-points over public networks. It is tunnelling of already encrypted traffic, governed by IPSec protocol, and maintains packet level encryption, making it almost impossible to identify the application running through end-points

can't make the classification just by looking at the IP packet header

- 'encrypted' VPN traffic is 3-fold, IPSec, Point-to-Point Tunnelling protocol, and SSL
- when VPN traffic is intercepted, the attacker will not be sble to identify the legitimate IP address of the end-points of the tunner, (source or destination)

## Related lit

- other classification done by analysis of port numbers, and works well, but no one is forced to use specific ports so the process isn't reliable
- Paxson (4) found that analysis of statistical feature like packet length, flow duration, inter-arrical times, in simple models can be good approximations
- Bernaille (5) use the first five packets of a stateful TCP connection
- Park et all (6) was similar to Bernaille, according to specific types of associated applications in order to achieve scalable QoS. Uses feature selection on ISPs and perform feature reduction to balance performance and complexity
- Erman (7) used ML clustering using unidirectional flow statistics. If found that is works better for server to client directio
- Williams (8) use NB similar to 7, used to identify application based on per-flow statistics derived from payload independent features such as packet lenght and inter-arrival time
- Nguyen and Armitage (9) use real-time NB, showed that classification is possible even when flow's beginning is lost/midway
- Crotti (10) used 'finger-printing' based on size, inter-arrical time and arrival order
- Bonfiglio (11) clasify skupe in real time used complimentary Bays Classifier and deep packet inspection. Leveraged randomness
- Paper uses supervised ML algos as either VPN encrypted or not

## Data

- goal was to perform binary classification of VPN vs non-VPN, rather than 8 label application category (Skype, browsing, etc)
- reorganized 8 data sets (4 time-based, each with a vpn and non-vpn version), into 7 datasets, one for each category (a new dataset was created by combining the vpn and non-vpn versions and for all time versions, partitioned by category)
- they removed empty and duplicate records (\*are we sure duplicate records are not valid records??)
- data was normalized, since the range of the different features varied greatly, they used z-score normalization (\*assumes normal distribution)
- vpn and non-vpn classes were balanced
- use 80/20 for training test, selected at random
- also says the selected the partition 'in a strategic fashion' to make the VPN and non-VPN records evenly distributed in in test/training data sets, \*\*how is this random??
- 'observation to variable ratio ... was 72', \*\*what's this ratio?
- LR - simple, fast, performs well on data sets that are linearly separable. Unless decision boundary is known to be degree > 1, first one to try
- SVM - identifies hyper-plane with maximum margin that separates space of input variables. Although by itself is still a line/plane, with kernel transform can be used in non-linear situations
- NB - probabilistic based on Bayes theorem (naive is the assumption that attributes are conditionally independent). Can have poor performance if this is violated (They used Gaussian)
- kNN - 'lazy' (doesn't actually do anything until prediction), measured on the majority vote of the classes of the nearest 'k' neighbours (distance can be euclidian, manhattan, etc. Paper doesn't say)
- RF - grows random decision trees based on random subset of data and subset features
- GBT - grows decision trees one after another guided by 'boosting'
- RF are robust and easier to tune, GBT perform better with less subtrees when well tuned
- LR, RBF, SVM, NB work well on simple linear hypothesis functions
- knn was chosen since it could have unique predictive performance even when othe rmodels don't work
- RF/GBT is for very complicated non-linear functions
- LR, RBF, SVM, NB are high-bias, while GBT and RF are high-variance
- didn't use deep learning since not enough data
- shallow artificial neural networks may work but often don't do as well as the models chosen (and ar eslow to train, hard to interpret, etc.)

## Model Selection

- used statistical metrics to optimize the model
- a cycle is -> train model, validate model, generate metrics.
- validation must be statistically independent of the training set to limit overfitting
- two common model evals: holdout and cross-validation
  - holdout - hold out an independent validation set (different from test data set, which is only used once in the fina ltest), from the original training data set. Bad, model evaluates on the same validation set in every iteration so tends to overfit the test data
  - cross-validation rotates training/test dataset, so each sample is used as a test at least once. They are averaged to be the evaluation metric.
- they used k-fold cross-validation
- accuracy = ratio of correct to total (TP + TN / Total). Best metric when classes are balanced and when + and - are both important
- precision = TP / (TP + FP) . The 'correctness' of the positive predictions. Metric to focus on when a high hit rate of + predictions is desirable
- sensitivity/recall = TP / (TP + FN). Correctness of how all actual positives are predicted. Most valuable when want the model to pick as many positive samples from actual positives
- specificity - TN / (FP + TN). Correctness of how all actual - are predicted. Counterpart to sensitivity on the - class, important when we want to pick as many TN

## Hyperparameter Optimization

- choosing a set of parameters outside the learning process, that helps to obtain the best outcome of a model. Could include grid search, random search, bayesian optimization, and gradient-based optimisation
- they used grid-search for RF and GBT models

## Feature Selection

- select subset of the most relevant/important attributes
- reducing makes the modle simpler and shortens training time, and could improve the generalisation by reducing overfitting.
- they do recursive feature elimination, wrapper-type feature selection: recursively constructs learning models with each feature excluded and examines the resulting validation metric without it. Feature with least importance that doesn't affect the validation metric much when removed, is dropped. Computationally expensive at higher dimensions, but works

## Bias-Variance Trade off

- hard to have model with high representation power on training data that is generalizable.
- underfitting = high-bias problem, caused by not enough data or high-bias model
  (model is too simple for the hypothesis). Need more data or need to switch models
- overfitting = high-variance problem, can limit by feature selection, hyperparameter-tuning, ensemble methods, and regularization (model dependent)
- solution is to optimize towards best cross-validated metric
  - if low (underfitted), change models or rework dataset
  - if good, but final evaluation on test data is low (overfitted), use control techniques

## Results

- best were GBT and RF
- KNN was okay,
- rest were inconsistent

## Model parameter search

- used grid search for hyper-parameters, chosen manually, following scikit learn
- number of base decision tree models was most important: smaller = simpler so faster, less change to overfit, more change to underfit.
- three other metrics: max tree depth, max features, min samples leaf
- for GBT, learning rate is the rate t oapply the gradient in each itration: larger = faster it will converge, but less accurate
- Ginin is the gini impurity versus entropy which calculates information gain. Gini is easier to calculate, but doesn't change the CV metrics much

- each of the 7 categories had different optimal parameters, but a general work-for-all set of parameters would simplify optimizations.
- Used voting from each of the 7 categories, to compute it

## Feature Selection

- RFE was used to get optimal amounts of features
- in RF and GBT models, RFE was compared to non-RFE, but it didn't increase results by a significant measure, so isn't considered worth it due to it's computational cost
- It is interesting that although a fairly large
  number of features was required to obtain the maximum accuracy, only a few features
  were necessary to obtain ‘good’ results (>90% accuracy as listed in Tables 8 and 9). This
  result can be considered useful in the scenarios when simplicity (e.g. when building rule
  based classifiers) and speed (e.g. in real-time classification) are of higher priority than
  accuracy.
- also tried PCA in TF model, but it actually reduced accuracy by 5%

## Results

- only RF and GBT where good enough to try for further enhancement
- poor accuracy and underfitting issues for LR, NB, SVM, and KNN)
- RF and GBT good without underfitting issues
- higher effectiveness of high-variance models suggests the the hypothesis function is not simple enough to be captured by simple models
- additional optimization of RF and GBT increased performance by a couple percent, but extra RFE feature selection didn't help (\*\*features identified by this could be used to develope fast-rule-based classification tools)
- confident that RF and GBT could be used to predict future data
- Some recommendations of future work would include collecting network traffic data
  using comprehensive cybersecurity defensive solutions like Intrusions Detection Systems,
  Firewalls and Honeynets and using similar machine-learning algorithms to classify the
  above-mentioned defensive solutions, using the acquired network traffic flow data sets.
