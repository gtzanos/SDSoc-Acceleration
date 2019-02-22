# Gaussian Naive Bayes

The Naive Bayes IP is used forNaive Bayes applications. It is produced with SDSoC 2016.3 version.

The code provided in this folder is a C-based implementation for NaiveBayes Training an d Prediction, optimised for ZED board, and is a case study of FPGA-Accelerated Machine Learning.


### Testing Logistic Regression example in Hardware
The C source files are provided here without project files, but they contain HLS/SDS directives specific to Xilinx SDSoC. 

`!The code of the hardware function is not fully annotated and contains only interface directives.!`

If you want to create a SDSoC project using these sources you may find the following instructions helpful:

1.  Launch SDSoC and create a new empty project. Choose `zed` as target platform.
1.  Add the C sources in `src/` and set `NBtraining_accel` or `NBprediction_accel` as hardware function. Set clock frequency at `100.00 MHz`.
1.  All design parameters are set in the file `src/accelerator.h`.
1.  Select `Generate Bitstream` and `Generate SD Card Image`.
1.  Run `SDRelease`.
	
#### Performance (`NBtraining_accel`, `2000 DataPack`)
Speedup (vs ARM-A9)	|	16.8
:----------|----------:
SW-only `ARM Cortex-A9 @ 666.67MHz` (Measured time)	|	273 msec
HW accelerated (Measured time)		|	16 msec


#### Resource utilization estimates for hardware accelerator
Resource	|	Used	|	Total	|	% Utilization
:----------:|----------:|----------:|:----------
DSP	|	197	|	220	|	89.55
BRAM	|	56	|	140	|	40
LUT	|	37929	|	53200	|	71.3
FF	|	29271	|	106400	|	27.51

#### Performance (`NBprediction_accel`, `2000 Examples`)
Speedup (vs ARM-A9)	|	14
:----------|----------:
SW-only `ARM Cortex-A9 @ 666.67MHz` (Measured time)	|	783 sec
HW accelerated (Measured time)		|	0.5 sec


#### Resource utilization estimates for hardware accelerator
Resource	|	Used	|	Total	|	% Utilization
:----------:|----------:|----------:|:----------
DSP	|	96	|	220	|	43.64
BRAM	|	56	|	140	|	40
LUT	|	28524	|	53200	|	53.62
FF	|	25251	|	106400	|	23.73
	
### Contacts
For any question or discussion, please contact the authors:

* Christoforos Kachris: kachris@microlab.ntua.gr
* Giorgos Tzanos: grg.tzan@gmail.com
