# SwinScale-LFVS
SWINSCALE-LFVS: PARALLEL FEATURE INTEGRATION FOR LIGHT FIELD VIEW SYNTHESIS
**Abstract** 
Light Field (LF) view synthesis aims to synthesize a dense set of LF views from a sparse set of input views. Although many recent learning-based methods have shown promising results in this task, they often rely on deep residual networks or on multiple LF representations to extract dense features, without fully exploiting the geometric structure of the LFs. In this paper, we introduce SwinScale-LFVS, a novel framework that combines the strengths of the Swin Transformer and the Multi-Scale Convolutional Network in parallel streams. The first stream uses a Swin Transformer to model local and global features using a geometry-aware Angular Mutual Self Attention (AMSA) network, and the second stream uses multi-scale 3D convolutions to extract dense features and to ensure spatial-angular consistency in synthesized LF views. The outputs from these streams are integrated and processed by an LF View Synthesis (LFVS) network to synthesize high-quality dense LF views. Extensive experiments show that SwinScale-LFVS outperforms existing methods on both real-world and synthetic datasets. 

<p align="center">
  <img src="SwinScale_LFVS.png" width="800"/>
  <br>
  <b>Fig. 1.</b> <i>SwinScale-LFVS modules: (a) Swin-Transformer Stream, (b) Multi-Scale Convolution Stream, and (c) LF View Synthesis..</i>
</p>

### 📊 Table I. Quantitative Comparison on Synthetic Datasets (PSNR / SSIM)

| Dataset   | LFASR-GEO [9]  | GA-MRVR [10]        | Deformable-LFVS [11]   | LF-EASR [12]         | Distg-ASR [13]       | <b>SwinScale-LFVS</b>      |
|-----------|----------------|----------------------|-------------------------|------------------------|------------------------|-----------------------------|
| HCI-new   | 32.29 / 0.911  | 34.28 / 0.936        | <u>34.52 / 0.936</u>    | 34.27 / <u>0.949</u>   | <u>34.16 / 0.969</u>   | <b>35.75 / 0.972</b>         |
| HCI-old   | 35.73 / 0.937  | 40.82 / 0.953        | 41.25 / 0.958           | <u>41.47 / 0.970</u>   | <b>42.13 / 0.974</b>   | <u>41.97 / 0.975</u>         |
| <b>Average</b> | 34.01 / 0.924  | 37.55 / 0.945        | 37.89 / 0.955           | 37.87 / <u>0.960</u>   | <u>38.14 / 0.972</u>   | <b>38.86 / 0.974</b>         |


### 📊 Table II. Quantitative Comparison on Real-World Datasets (PSNR / SSIM)

| Dataset     | LFASR-GEO [9] | GA-MRVR [10]     | Deformable-LFVS [11] | LF-EASR [12]     | Distg-ASR [13]     | **SwinScale-LFVS** |
|-------------|----------------|------------------|-----------------------|------------------|--------------------|--------------------|
| UCSD        | 40.78 / 0.982  | _42.91 / 0.987_  | _43.19 / 0.987_       | 43.15 / 0.980    | _43.60 / 0.986_    | **43.87 / 0.987**  |
| Occlusions  | 36.43 / 0.971  | _39.04 / 0.981_  | _39.45 / 0.984_       | _39.18 / 0.979_  | _39.40 / 0.982_    | **39.85 / 0.983**  |
| Reflective  | 36.25 / 0.945  | _39.04 / 0.962_  | _39.19 / 0.963_       | _38.91 / 0.958_  | _39.02 / 0.960_    | **39.45 / 0.962**  |
| **Average** | 37.82 / 0.966  | _40.33 / 0.976_  | _40.61 / 0.978_       | _40.41 / 0.972_  | _40.67 / 0.976_    | **41.06 / 0.977**  |
