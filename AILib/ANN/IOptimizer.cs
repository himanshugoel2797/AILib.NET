using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math;

namespace AILib.ANN
{
    public interface IOptimizer
    {
        (Matrix[], Vector[]) Optimize(Matrix[] w, Vector[] b, Matrix[] nabla_w, Vector[] nabla_b, float rate);
    }
}
