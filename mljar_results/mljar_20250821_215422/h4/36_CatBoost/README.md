# Summary of 36_CatBoost

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.1
- **depth**: 8
- **rsm**: 0.8
- **loss_function**: RMSE
- **eval_metric**: MAE
- **explain_level**: 0

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: False

## Optimized metric
mae

## Training time

10.1 seconds

### Metric details:
| Metric   |    Score |
|:---------|---------:|
| MAE      | 0.808069 |
| MSE      | 1.19842  |
| RMSE     | 1.09472  |
| R2       | 0.185751 |
| MAPE     | 0.260581 |



## Learning curves
![Learning curves](learning_curves.png)
## True vs Predicted

![True vs Predicted](true_vs_predicted.png)


## Predicted vs Residuals

![Predicted vs Residuals](predicted_vs_residuals.png)



[<< Go back](../README.md)
