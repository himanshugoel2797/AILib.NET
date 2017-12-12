using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math.GPU;

namespace AILib.ANN.ActivationFunctions
{
    public class Sigmoid : IActivationFunction
    {
        public float Activation(float o)
        {
            return (1.0f / (1.0f + (float)System.Math.Exp(-o)));
        }

        public float DerivActivation(float o)
        {
            return Activation(o) * (1 - Activation(o));
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
