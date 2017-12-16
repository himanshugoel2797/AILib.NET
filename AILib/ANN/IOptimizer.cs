using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if CPUMATH
using AILib.Math;
#else
using AILib.Math.GPU;
#endif
namespace AILib.ANN
{
    public interface IOptimizer
    {
        (Matrix[], Vector[]) Optimize(Matrix[] w, Vector[] b, Matrix[] nabla_w, Vector[] nabla_b, float rate);
    }
}
