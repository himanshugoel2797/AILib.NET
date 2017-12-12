using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AILib.Math.GPU
{
    public class Matrix
    {
        private static Dictionary<string, ComputeKernel> kernels = new Dictionary<string, ComputeKernel>();
        private static ComputeProgram prog;

        private MemoryBuffer buffer;
        private float[] data;
        private bool dirty;
        private bool pullFromGPU;

        public int Width { get; private set; }
        public int Height { get; private set; }

        public Matrix(int w, int h)
        {
            Width = w;
            Height = h;
            data = new float[w * h];

            buffer = Accelerator.CreateMemory(w * h * sizeof(float), true, true);
            dirty = true;

            if (prog == null)
            {
                prog = Accelerator.CreateProgram(File.ReadAllText("OpenCL/matrix.cl"));
                kernels["matrixMul"] = Accelerator.CreateKernel("matrixMul", prog);
            }
        }

        internal void updateMatrix()
        {
            if (dirty)
            {
                Accelerator.SubmitMemoryWrite(buffer, 0, Width * Height * sizeof(float), data);
                dirty = false;
            }else if(pullFromGPU)
            {
                Accelerator.SubmitMemoryRead(buffer, 0, Width * Height * sizeof(float), data);
                pullFromGPU = false;
            }
        }

        public float this[int x, int y]
        {
            get
            {
                if (pullFromGPU | dirty)
                {
                    updateMatrix();
                    Accelerator.Barrier();
                }

                return data[Height * x + y];
            }
            set
            {
                if (pullFromGPU)
                {
                    updateMatrix();
                    Accelerator.Barrier();
                }

                data[Height * x + y] = value;
                dirty = true;
            }
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            Matrix c = new Matrix(b.Width, a.Height);

            //var tple = new Tuple<int, int, int, int>(a.Width, a.Height, b.Width, b.Height);
            //if (!matrixProgs.ContainsKey(tple))
            //{
            //Initialize program
            //}
            //var prog = matrixProgs[tple];

            //Setup the program and call it
            bool wereDirty = (a.dirty | b.dirty);
            if (a.dirty) a.updateMatrix();
            if (b.dirty) b.updateMatrix();

            if (wereDirty) Accelerator.Barrier();

            c.pullFromGPU = true;
            c.dirty = false;

            var k = kernels["matrixMul"];
            k.SetArgument(0, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, Accelerator.intPtrSize, b.buffer.mem);
            k.SetArgument(3, sizeof(int), a.Height);
            k.SetArgument(4, sizeof(int), b.Width);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });

            return c;
        }

        public static Vector operator *(Matrix a, Vector b)
        {
            Vector c = new Vector(a.Height);

            //var tple = new Tuple<int, int, int, int>(a.Width, a.Height, b.Width, b.Height);
            //if (!matrixProgs.ContainsKey(tple))
            //{
            //Initialize program
            //}
            //var prog = matrixProgs[tple];

            //Setup the program and call it
            bool wereDirty = (a.dirty | b.dirty);
            if (a.dirty) a.updateMatrix();
            if (b.dirty) b.updateMatrix();

            if (wereDirty) Accelerator.Barrier();

            c.pullFromGPU = true;
            c.dirty = false;

            var k = kernels["matrixMul"];
            k.SetArgument(0, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, Accelerator.intPtrSize, b.buffer.mem);
            k.SetArgument(3, sizeof(int), a.Height);
            k.SetArgument(4, sizeof(int), 1);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });

            return c;
        }
    }
}
