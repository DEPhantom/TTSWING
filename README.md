# TTSWING: a Dataset for Table Tennis Swing and Racket Kinematics Analysis

Che-Yu Chou, Zheng-Hao Chen, Yung-Hoh Sheu, Hung-Hsuan Chen, Min-Te Sun, Sheng K. Wu

The repository contains the TTSWING dataset, a novel dataset for table tennis swing analysis. It also includes the code for the experiments on the dataset.

## Dataset
The dataset comprises comprehensive swing information obtained through 9-axis sensors integrated into custom-made racket grips, accompanied by anonymized demographic data of the players. The dataset contains 90,000+ collected stroke data from 93 Taiwanese players in Group A (the elite group). The participants included 53 males and 40 females; their ages ranged between 13 and 28 years old; their years of experience ranged between 1 to 19 years. They were asked to perform at least one of three different swing modes: swinging in the air, full power stroke, and stable hitting, denoted as mode 0 to mode 2, respectively. The participants swing the racket 50 times continuously in each mode to generate a complete waveform set, which is further transformed into tabular data. You can download the dataset [here](https://github.com/DEPhantom/TTSWING/blob/main/dataset/TTSWING.csv)

> **Note**
> In order to safeguard players' privacy, we categorized player age, height, weight, and play years into three tiers ("high," "medium," and "low") based on the current data distribution. The dataset will be continually updated in the presence of additional data or when the features can be released publicly.

<details>
  <summary><b>Detail of the attributes</b></summary>
  
  The unit of the accelerations (e.g., ax_mean, ay_mean, az_mean) is LSB/G (least significant bit per unit of G-force). By multiplying this value by 2/32768, the original G value can be obtained. The unit of angular velocities (e.g., gx_mean, gy_mean, gz_mean) is LSB/deg/s (least significant bit per unit of angular velocity). By multiplying this value by 250/32768, the original DPS (degree per second) can be obtained.
  
  | Field              | Description |
  |--------------------|-------------|
  | id                 | An unique ID to identify players |
  | date               | The data collected date|
  | testmode           | Three testing modes: swing in the air, full power stroke, and stable hitting, with values 0, 1, and 2 |
  | teststage          | This value is only useful when testmode is 1. The values 1 to 3 represent three different ball speeds served by the serving machine |
  | fileindex          | The round that the player performs the swing |
  | count              | The count of swings in this round |
  | ax_mean            | The mean of x-axis acceleration (unit: LSB/G) |
  | ay_mean            | The mean of y-axis acceleration (unit: LSB/G) |
  | az_mean            | The mean of z-axis acceleration (unit: LSB/G) |
  | gx_mean            | The mean of x-axis angular velocity (unit: LSB/deg/s) |
  | gy_mean            | The mean of y-axis angular velocity (unit: LSB/deg/s) |
  | gz_mean            | The mean of z-axis angular velocity (unit: LSB/deg/s) |
  | ax_var             | The variance of x-axis acceleration |
  | ay_var             | The variance of y-axis acceleration |
  | az_var             | The variance of z-axis acceleration |
  | gx_var             | The variance of x-axis angular velocity |
  | gy_var             | The variance of y-axis angular velocity |
  | gz_var             | The variance of z-axis angular velocity |
  | ax_rms             | The root mean square of x-axis acceleration |
  | ay_rms             | The root mean square of y-axis acceleration |
  | az_rms             | The root mean square of z-axis acceleration |
  | gx_rms             | The root mean square of x-axis angular velocity |
  | gy_rms             | The root mean square of y-axis angular velocity |
  | gz_rms             | The root mean square of z-axis angular velocity |
  | a_max              | The maximum acceleration of a swing |
  | a_mean             | The mean of acceleration of a swing |
  | a_min              | The minimum acceleration of a swing |
  | g_max              | The maximum angular velocity of a swing |
  | g_mean             | The mean angular velocity of a swing |
  | g_min              | The minimum angular velocity of a swing |
  | a_fft              | The Fourier transform of the acceleration |
  | g_fft              | The Fourier transform of the angular velocity |
  | a_psdx             | The power spectral density of the acceleration |
  | g_psdx             | The power spectral density of the angular velocity |
  | a_kurt             | The kurtosis of the acceleration |
  | g_kurt             | The kurtosis of the angular velocity |
  | a_skewn            | The skewness of the acceleration |
  | g_skewn            | The skewness of the angular velocity |
  | a_entropy          | The spectral entropy of the acceleration |
  | g_entropy          | The spectral entropy of the angular velocity |
  | gender             | The gender of the player: 1 for males and 0 for females. |
  | age                | The age of the player |
  | play years         | Number of years players have played ball games |
  | height             | The height of the player |
  | weight             | The weight of the player |
  | handedness         | Player’s dominant hand: 1 for the right hand; 0 for the left hand |
  | hold racket handed | The hand holds the racket: 1 for the right hand and 0 for the left hand |
  
</details>

## Files

- `classification_gender.py` - The experiment in predicting a player's gender.
- `classification_age.py` - The experiment in predicting a player's age.
- `classification_mode.py` - The experiment in predicting testmode.
- `classification_holding.py` - The experiment in predicting a player's racket-holding hand.
- `classification_exp_years.py` - The experiment in predicting a player's years of experience.
- `general_utils.py` - Some general funcitons used in classification experiments.
- `globals.py` - Global variables for recording experiment results.

## Usage
To execute the code and run experiments, follow these.
Install
```Shell
git clone https://github.com/DEPhantom/TTSWING.git
```
Then `cd` into the root and run the command:
```Python
python classification_<X>.py
```
where `<X>` could be `gender`, `age`, `mode`, `holding`, or `exp_years`.

Each script generates an Excel file to summarize the results.

To compute statistical summaries of numerical and categorical features, run:
```Python
python show_stat.py
```

It outputs `categorical_stat_summary.csv` and `numerical_stat_summary.csv` in the same folder.

## Requirements
We tested the codes on `Python 3.10.4` (installed by `Conda`) on Mac. Package versions are available in `requirements.txt`, which is generated via `pip freeze > requirements.txt`. To install these packages, runs

```sh
pip install -r requirements.txt
```
