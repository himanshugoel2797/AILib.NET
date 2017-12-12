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

            if (prog == null)
            {
                prog = Accelerator.CreateProgram(File.ReadAllText("OpenCL/vector.cl"));
                //kernels["matrixMul"] = Accelerator.CreateKernel("matrixMul", prog);
            }
        }

        internal void updateMatrix()
        {
            if (dirty)
            {
                Accelerator.SubmitMemoryWrite(buffer, 0, Length * sizeof(float), data);
                dirty = false;
            }
            else if (pullFromGPU)
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
                    Accelerator.Barrier();
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
    }
}
