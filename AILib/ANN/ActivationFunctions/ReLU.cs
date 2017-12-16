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
    public class ReLU : IActivationFunction
    {
        public float Activation(float o)
        {
            if (o >= 0)
                return o;

            return 0;
        }

        public float DerivActivation(float o)
        {
            if (o >= 0)
                return 1;
            return 0;
        }

        public Vector Activation(Vector o)
        {
            Vector n = new Vector(o.Length);
            for (int i = 0; i < n.Length; i++)
                n[i] = Activation(o[i]);
            return n;
        }

        public Vector DerivActivation(Vector o)
        {
            Vector n = new Vector(o.Length);
            for (int i = 0; i < n.Length; i++)
                n[i] = DerivActivation(o[i]);
            return n;
        }
    }
}
