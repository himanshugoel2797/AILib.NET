using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AILib
{
    public class CoreLibrary
    {
        public static void Initialize()
        {
            AILib.Math.GPU.Accelerator.Initialize();
        }
    }
}
