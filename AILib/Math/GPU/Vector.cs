using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AILib.Math.GPU
{
    public class Vector
    {
        private static Dictionary<string, ComputeKernel> kernels = new Dictionary<string, ComputeKernel>();
        private static ComputeProgram prog;

        internal MemoryBuffer buffer;
        internal float[] data;
        internal bool dirty;
        internal bool pullFromGPU;

        public int Length { get; private set; }

        public Vector(int l)
        {
            Length = l;
            data = new float[l];

            buffer = Accelerator.CreateMemory(l * sizeof(float), true, true);
            dirty = true;
            progInit();
        }

        public Vector(float[] l)
        {
            Length = l.Length;
            data = l;

            buffer = Accelerator.CreateMemory(l.Length * sizeof(float), true, true);
            dirty = true;
            progInit();
        }

        private void progInit()
        {
            if (prog == null)
            {
                prog = Accelerator.CreateProgram(File.ReadAllText("OpenCL/vector.cl"));
                kernels["msub"] = Accelerator.CreateKernel("msub", prog);
            }
        }

        internal void updateMatrix()
        {
            if (dirty)
            {
                Accelerator.SubmitMemoryWrite(buffer, 0, Length * sizeof(float), data);
                dirty = false;
            }

            if (pullFromGPU)
            {
                Accelerator.SubmitMemoryRead(buffer, 0, Length * sizeof(float), data);
                pullFromGPU = false;
            }
        }

        public float this[int x]
        {
            get
            {
                if (pullFromGPU | dirty)
                {
                    updateMatrix();
                    Accelerator.WaitFinish();
                }

                return data[x];
            }
            set
            {
                if (pullFromGPU)
                {
                    updateMatrix();
                    Accelerator.Barrier();
                }

                data[x] = value;
                dirty = true;
            }
        }

        public static implicit operator float[](Vector v)
        {
            return v.data;
        }

        public override string ToString()
        {
            string r = "[";
            for(int i = 0; i < Length; i++)
            {
                r += this[i];
                if (i < Length - 1)
                    r += ", ";
            }

            return r + "]";
        }

        public static void MSub(Vector a, float b, ref Vector c)
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
            k.SetArgument(3, sizeof(int), a.Length);

            Accelerator.SubmitProgram(k, 2, null, new[] { (IntPtr)1024, (IntPtr)1024 }, new[] { (IntPtr)16, (IntPtr)16 });
        }
    }
}
