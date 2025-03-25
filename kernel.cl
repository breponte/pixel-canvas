__kernel void sawtoothKernel(__global float* X)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int stride = get_global_size(0);
    if (get_local_id(0) >= get_local_id(1)) {
        X[y * stride + x] = 1.0f;
    } else {
        X[y * stride + x] = 0.0f;
    }
}
