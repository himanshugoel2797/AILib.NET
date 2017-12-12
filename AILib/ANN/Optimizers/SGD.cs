using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math.GPU;

namespace AILib.ANN.Optimizers
{
    public class SGD : IOptimizer
    {
        public (Matrix[], Vector[]) Optimize(Matrix[] w, Vector[] b, Matrix[] nabla_w, Vector[] nabla_b, float rate)
        {
            for (int i = 1; i < w.Length; i++)
            {
                Vector.MSub(nabla_b[i], rate, ref b[i]);
                Matrix.MSub(nabla_w[i], rate, ref w[i]);
            }

            return (w, b);
        }
    }
}
