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
                kernels["msub"] = Accelerator.CreateKernel("msub", prog);
                kernels["madd"] = Accelerator.CreateKernel("madd", prog);
                kernels["vecvecmat_mult"] = Accelerator.CreateKernel("vecvecmat_mult", prog);
                kernels["trans_multProduct"] = Accelerator.CreateKernel("trans_multProduct", prog);
            }
        }

        internal void updateMatrix()
        {
            if (dirty)
            {
                Accelerator.SubmitMemoryWrite(buffer, 0, Width * Height * sizeof(float), data);
                dirty = false;
            }

            if (pullFromGPU)
            {
                Accelerator.SubmitMemoryRead(buffer, 0, Width * Height * sizeof(float), data);
                pullFromGPU = false;
            }
        }


        public override string ToString()
        {
            string r = "";
            for (int i = 0; i < Height; i++)
            {
                r += "[";
                for (int j = 0; j < Width; j++)
                {
                    r += this[j, i];
                    if (j < Width - 1)
                        r += ", ";
                }
                r += "]\n";
            }

            return r;
        }

        public static void TransposedMultiply(Matrix a, Vector b, Vector c, ref Vector d)
        {
            //Setup the program and call it
            bool wereDirty = (a.dirty | b.dirty | c.dirty);
            if (a.dirty) a.updateMatrix();
            if (b.dirty) b.updateMatrix();
            if (c.dirty) c.updateMatrix();

            if (wereDirty) Accelerator.WaitFinish();

            d.pullFromGPU = true;
            d.dirty = false;

            var k = kernels["trans_multProduct"];
            k.SetArgument(0, Accelerator.intPtrSize, d.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, Accelerator.intPtrSize, b.buffer.mem);
            k.SetArgument(3, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(4, sizeof(int), a.Height);
            k.SetArgument(5, sizeof(int), 1);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });
        }

        public float this[int x, int y]
        {
            get
            {
                if (pullFromGPU | dirty)
                {
                    updateMatrix();
                    Accelerator.WaitFinish();
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

        public static void MultiplyToMatrix(Vector a, Vector b, ref Matrix c)
        {
            bool wereDirty = (a.dirty | b.dirty);
            if (a.dirty) a.updateMatrix();
            if (b.dirty) b.updateMatrix();

            if (wereDirty) Accelerator.WaitFinish();

            c.pullFromGPU = true;
            c.dirty = false;

            var k = kernels["vecvecmat_mult"];
            k.SetArgument(0, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, Accelerator.intPtrSize, b.buffer.mem);
            k.SetArgument(3, sizeof(int), a.Length);
            k.SetArgument(4, sizeof(int), b.Length);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });
        }

        public static void Multiply(Matrix a, Matrix b, ref Matrix c)
        {
            //Matrix c = new Matrix(b.Width, a.Height);

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

            if (wereDirty) Accelerator.WaitFinish();

            c.pullFromGPU = true;
            c.dirty = false;

            var k = kernels["matrixMul"];
            k.SetArgument(0, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, Accelerator.intPtrSize, b.buffer.mem);
            k.SetArgument(3, sizeof(int), a.Height);
            k.SetArgument(4, sizeof(int), b.Width);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });
        }

        public static void Madd(Matrix a, Vector b, Vector c, ref Vector d)
        {
            //Setup the program and call it
            bool wereDirty = (a.dirty | b.dirty | c.dirty);
            if (a.dirty) a.updateMatrix();
            if (b.dirty) b.updateMatrix();
            if (c.dirty) c.updateMatrix();

            if (wereDirty) Accelerator.WaitFinish();

            d.pullFromGPU = true;
            d.dirty = false;

            var k = kernels["madd"];
            k.SetArgument(0, Accelerator.intPtrSize, d.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, Accelerator.intPtrSize, b.buffer.mem);
            k.SetArgument(3, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(4, sizeof(int), a.Height);
            k.SetArgument(5, sizeof(int), 1);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });
        }

        public static void Multiply(Matrix a, Vector b, ref Vector c)
        {
            //Setup the program and call it
            bool wereDirty = (a.dirty | b.dirty);
            if (a.dirty) a.updateMatrix();
            if (b.dirty) b.updateMatrix();

            if (wereDirty) Accelerator.WaitFinish();

            c.pullFromGPU = true;
            c.dirty = false;

            var k = kernels["matrixMul"];
            k.SetArgument(0, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, Accelerator.intPtrSize, b.buffer.mem);
            k.SetArgument(3, sizeof(int), a.Height);
            k.SetArgument(4, sizeof(int), 1);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });
        }

        public static void MSub(Matrix a, float b, ref Matrix c)
        {
            //Setup the program and call it
            bool wereDirty = (a.dirty | c.dirty);
            if (a.dirty) a.updateMatrix();
            if (c.dirty) c.updateMatrix();

            if (wereDirty) Accelerator.WaitFinish();

            c.pullFromGPU = true;
            c.dirty = false;

            var k = kernels["msub"];
            k.SetArgument(0, Accelerator.intPtrSize, c.buffer.mem);
            k.SetArgument(1, Accelerator.intPtrSize, a.buffer.mem);
            k.SetArgument(2, sizeof(float), b);
            k.SetArgument(3, sizeof(int), a.Height);
            k.SetArgument(4, sizeof(int), a.Width);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });
        }
    }
}
