using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math.GPU;

namespace AILib.ANN
{
    public interface ILossFunction
    {
        Vector Loss(Vector output, Vector expectedOutput);
        Vector LossDeriv(NeuralNetwork net, Vector output, Vector logit, Vector expectedOutput);
    }
}
