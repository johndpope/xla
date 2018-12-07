# How To Build And Run PyTorch For TPU

To build:

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Checkout the following commit ID:

  ```
  git checkout 75a2d8e2de4a73e16c3ea22f781673ea3e15a1f9
  ```

* Clone the _PyTorch/XLA_ repo:

  ```
  git clone --recursive https://github.com/pytorch/xla.git
  ```

* Apply the `pytorch.patch` to the current `xla` code. From within the _pytorch_ source folder:

  ```
  patch -p1 < xla/pytorch.patch
  ```

* Currently _PyTorch_ does not build with GCC 8.x. A known working GCC version is 7.3.x, so install that in your VM:

  ```
  apt-get install gcc-7 g++-7
  export CC=gcc-7
  export CXX=g++-7
  ```
  
* Build _PyTorch_ from source following the regular [instructions](https://github.com/pytorch/pytorch#from-source).

  ```
  python setup.py install
  ```

* Build the _PyTorch/XLA_ source:

  ```
  cd xla/
  python setup.py install
  ```

To run the tests, follow __one__ of the options below:

* Run on CPU using the XLA local client:

  ```
  export XLA_USE_XRT=0 export XLA_GRPC_HOST="" XLA_PLATFORM="CPU"
  ```

* Run on local CPU using the XRT client:

  ```
  export XLA_USE_XRT=1 XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
  ```
  
  Select any free TCP port you prefer instead of 40934 (totally arbitrary).

* Run on Cloud TPU using the XRT client, use one of the following:

  - Set the XRT_TPU_CONFIG environment variable:
  
    ```
    export XRT_TPU_CONFIG="tpu_worker;0;<IP of the TPU node>:8470"
    ```

  - Create a `$HOME/.pytorch_tpu.conf` file with the following content: `worker: tpu_worker <IP of the TPU node>:8470`


Note that the IP of the TPU node can change if the TPU node is reset. If _PyTorch_
seem to hang at startup, verify that the IP of your TPU node is still the same of
the one you have configured.


Then run `python test/test_operations.py`. Some of the tests are currently skipped.
