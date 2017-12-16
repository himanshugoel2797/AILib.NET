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

namespace AILib.ANN.ActivationFunctions
{
    public class ScaledSoftStep : IActivationFunction
    {
        public Vector Activation(Vector o)
        {
            //Vector a = (1 + Vector.Abs(o));
            //return o / (a * 2) + 0.5f;
            return null;
        }

        public Vector DerivActivation(Vector o)
        {
            //Vector tmp = (1 + Vector.Abs(o));
            //return 1 / (tmp * tmp * 2);
            return null;
        }
    }
}
