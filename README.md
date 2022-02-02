# membership_inference
This repository provides framework for testing ml-models for vulnerability to membership inference attack

Supported models:
- keras neural networks
- tensorflow neural networks
- sklearn models

There are 4 different libs for 4 different types of target models:
- estimators_bin_nn - for neural networks of binary classification
- estimators_bin_skl - for sklearn binary classificators
- estimators_multi_nn - for multiclass neural networks
- estimators_multi_skl - for multiclass sklearn-models

To use this framework you just need to pull membership_inference/MembershipInferenceTest and import MembershipInferenceTest

You can see tool usage examples in membership_inference/experiments
