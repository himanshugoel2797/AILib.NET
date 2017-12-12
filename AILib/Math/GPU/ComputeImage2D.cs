using OpenCL.Net;

namespace AILib.Math.GPU
{
    public class ComputeImage2D
    {
        internal IMem img;

        public int Height { get; internal set; }
        public int Width { get; internal set; }
    }
}