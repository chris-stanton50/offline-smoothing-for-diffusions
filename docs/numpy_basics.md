# Numpy basics

- Creating a numpy array: `np.ones` `np.eye` `np.zeros` `np.empty`, `np.arange` `np.linspace` etc. Can specify the `dtype` as a `kwarg` to these functions. The default dtype is `np.float64`
- Can sort and concatenate arrays with `np.sort` and `np.concatenate`. Check the documentation when needed to figure out what these mean.

To do: 

If we were to define a new distribution to represent higher order numerical schemes, we would need to define the `rvs` method of the new distribution.

This method would need to have a `size` kwarg. 
 
 Note: the `b` function in the MvSDE must input (1, dimX) ndarray and output (1, dimX) ndarray.
 The function should also appropriately broadcast, so that an input of (N, dimX) ndarray outputs a (n, dimX) ndarray, such that the function `b` has been applied individually to each of the n  (1, dimX) dimensional vectors that are given as inputs.


 Note: the `sigma` function in the MvSDE must input (1, dimX) ndarray and output (dimX, dimX) ndarray.
 The function should also appropriately broadcast, so that an input of (N, dimX) ndarray outputs an (N, dimX, dimX) ndarray, such that the function `sigma` has been applied individually to each of the n  (1, dimX) dimensional vectors that are given as inputs.

