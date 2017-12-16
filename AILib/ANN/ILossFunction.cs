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
    public interface ILossFunction
    {
        Vector Loss(Vector output, Vector expectedOutput);
        Vector LossDeriv(NeuralNetwork net, Vector output, Vector logit, Vector expectedOutput);
    }
}
