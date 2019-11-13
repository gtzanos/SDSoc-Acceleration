# MiniKKR with StarPU 

## Build

The Makefile in source directory should be used in order to build the program. 
The way of compiling the project is similar to the native application plus the use of StarPu libraries.

## Results

The application has been tested in 2 different systems:

-> 6-core CPU: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
    
    * Native Applcation : 17 sec
    * With StarPU : 18 sec

-> 40-core CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz ( 2 Sockets , 20 cores / Socket )

    * Native Application : 0.5 sec
    * With StarPU : 5 sec
