# ([Time Series Domain Adaptation via Sparse Associative Structure Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/16846))[AAAI2021]

## Requirements

- Python  3.7

- Pytorch 1.7

  

## Quick Start

The pytorch version of sasa basically replicated the results of the tensorflow version of sasa on the Boiler dataset.

You can run it with the following command.

```
python  main.py
```



## Experiment Result

AUC scores on the Boiler data set.

|  1->2  |  1->3  |  2>1   |  2->3  |  3->1  |  3->2  | mean   |
| :----: | :----: | :----: | :----: | :----: | :----: | ------ |
| 0.7260 | 0.9599 | 0.8575 | 0.9451 | 0.9175 | 0.6835 | 0.8483 |

