<div align="center">
  <img src=""><br><br>
</div>

# TTSWING: a Dataset for Table Tennis Swing Analysis

Che-Yu Chou, Zheng-Hao Chen, Yung-Hoh Sheu, Hung-Hsuan Chen, Sheng K. Wu

The repository contains TTSWING, a novel dataset for table tennis swing analysis, and codes which include four classification and one regression experiments for the work on ["TTSWING: a Dataset for Table Tennis Swing Analysis"](https://arxiv.org/abs/2306.17550) from IJCAI 2023 workshop ([Intelligent Technologies for Precision Sports Science](https://wasn.csie.ncu.edu.tw/workshop/IT4PSS.html)).

## Citation
If you used this dataset in your research and want to cite it here is how you do it:
```
@article{chou2023ttswing,
  title={TTSWING: a Dataset for Table Tennis Swing Analysis},
  author={Chou, Che-Yu and Chen, Zheng-Hao and Sheu, Yung-Hoh and Chen, Hung-Hsuan and Wu, Sheng K},
  journal={arXiv preprint arXiv:2306.17550},
  year={2023}
}
```

## Dataset
The dataset comprises comprehensive swing information obtained through 9-axis sensors integrated into custom-made racket grips, accompanied by anonymized demographic data of the players. The dataset contain more than 90,000 stroke data have been collected. There are current 93 Taiwanese players from Group A participate in the collection process, including 53 males and 40 females. The participants’ age and playing experience ranged widely, from 13 to 28 years old and 1 to 19 years of experience. They were asked to perform at least one of three different swing modes: swinging in the air, full power stroke, and stable hitting , denoted as mode 0 to mode 2, respectively. Each mode requires the participants to swing the racket 50 times continuously to generate a complete waveform set and transform as the tabular data. [You can download here](https://github.com/DEPhantom/DART_project/tree/main/Code/dataset)

> **Note**
> In order to safeguard player privacy, we categorized player age, height, weight, and play years into three tiers based on the current data distribution: "high," "medium," and "low." The dataset will be continually updated in the presence of additional data or when the features can be made public.

<details>
  <summary><b>Detail of the attributes</b></summary>
  
  | Field              | Description |
  |--------------------|-------------|
  | id                 | A number used to identify players |
  | date               | The date when the data was collected |
  | testmode           | Three mode for swing in the air, full power stroke and stable hitting, respectively |
  | teststage          | This value is only useful when testmode is 1. The value 1 to 3 represent three different ball speeds set by the serving machine |
  | fileindex          | The round that the player perform the swing |
  | count              | The number of swings in each round |
  | ax_mean            | Average value of x-axis acceleration |
  | ay_mean            | Average value of y-axis acceleration |
  | az_mean            | Average value of z-axis acceleration |
  | gx_mean            | Average value of x-axis angular velocity |
  | gy_mean            | Average value of y-axis angular velocity |
  | gz_mean            | Average value of z-axis angular velocity |
  | ax_var             | The variance of x-axis acceleration |
  | ay_var             | The variance of y-axis acceleration |
  | az_var             | The variance of z-axis acceleration |
  | gx_var             | The variance of x-axis angular velocity |
  | gy_var             | The variance of y-axis angular velocity |
  | gz_var             | The variance of z-axis angular velocity |
  | ax_rms             | The root mean sqare error of x-axis acceleration |
  | ay_rms             | The root mean sqare error of y-axis acceleration |
  | az_rms             | The root mean sqare error of z-axis acceleration |
  | gx_rms             | The root mean sqare error of x-axis angular velocity |
  | gy_rms             | The root mean sqare error of y-axis angular velocity |
  | gz_rms             | The root mean sqare error of z-axis angular velocity |
  | a_max              | The maximum value of the square root of the acceleration per swing |
  | a_mean             | Average of square root of acceleration per swing |
  | a_min              | Minimum value of square root of acceleration per swing |
  | g_max              | The maximum value of the square root of the angular velocity in each swing |
  | g_mean             | The average of the square root of the angular velocity in each swing |
  | g_min              | The minimum value of the square root of the angular velocity in each swing |
  | a_fft              | The fourier transform of the acceleration |
  | g_fft              | The fourier transform of the angular velocity |
  | a_psdx             | The power spectral density of the acceleration |
  | g_psdx             | The power spectral density of the angular velocity |
  | a_kurt             | The kurtosis of the acceleration |
  | g_kurt             | The kurtosis of the angular velocity |
  | a_skewn            | The skewness of the acceleration |
  | g_skewn            | The skewness of the angular velocity |
  | a_entropy          | The spectral entropy of the acceleration |
  | g_entropy          | The spectral entropy of the angular velocity |
  | gender             | The gender of the player. 1 for male and 0 for female. |
  | age                | The age of the player |
  | play years         | Number of years players have played ball games |
  | height             | The height of the player |
  | weight             | The weight of the player |
  | handedness         | Player’s dominant hand. The value 1 for right hand and the valu 0 for left hand |
  | hold racket handed | The hand holds the racket. The value 1 for right hand and the valu 0 for left hand |
  
</details>

## Files

- `classification_gender.py` - The experiment in predicting player's gender.
- `classification_age.py` - The experiment in predicting player's age.
- `classification_mode.py` - The experiment in predicting testmode.
- `classification_holding.py` - The experiment in predicting player's racket-holding hand.
- `regression_exp_years.py` - The experiment in predicting player's years of experience.
- `general_utils.py` - Some general funcitons used in classification experiments.
- `globals.py` - Global variables for recording experiment result.

## Usage
To execute the code and run experiments, follow these.
Install
```Shell
git clone https://github.com/DEPhantom/TTSWING.git
```
Then `cd` into the root and run the command:
```Python
python classification_gender.py
```
There is a output file recording experiment result.

## Requirements
If you want to reproduce experiments, please follow these requirements.
* python==3.9.13
* numpy==1.22.3
* pandas==1.4.2
* sklearn==1.0.2
* tensorflow==2.8.3
* keras==2.8.0
* matplotlib==3.5.2
* openpyxl==3.0.10
* tqdm==4.65.0
