using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math;

namespace AILib.ANN.ActivationFunctions
{
    public class Tanh : IActivationFunction
    {
        public float Activation(float o)
        {
            return (float)System.Math.Tanh(o);
        }

        public float DerivActivation(float o)
        {
            return (float)(1.0d - System.Math.Pow(System.Math.Tanh(o), 2));
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
