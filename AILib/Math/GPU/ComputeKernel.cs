using OpenCL.Net;

namespace AILib.Math.GPU
{
    public class ComputeKernel
    {
        internal Kernel kernel;

        public void SetArgument(uint i, int size, object val)
        {
            Cl.SetKernelArg(kernel, i, (IntPtr)size, val);
        }
    }
}