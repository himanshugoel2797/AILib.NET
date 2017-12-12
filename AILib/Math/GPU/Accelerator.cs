using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace AILib.Math.GPU
{
    public class Accelerator
    {
        private static Context _context;
        private static Device _device;
        private static CommandQueue _queue;

        internal static int intPtrSize;

        private static void ContextNotify(string errInfo, byte[] data, IntPtr cb, IntPtr userData)
        {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        private static void Setup()
        {
            intPtrSize = Marshal.SizeOf(typeof(IntPtr));

            ErrorCode error;
            Platform[] platforms = Cl.GetPlatformIDs(out error);
            List<Device> devicesList = new List<Device>();

            foreach (Platform platform in platforms)
            {
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out error).ToString();
                Console.WriteLine("Platform: " + platformName);
                //We will be looking only for GPU devices
                foreach (Device device in Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error))
                {
                    Console.WriteLine("Device: " + device.ToString());
                    devicesList.Add(device);
                }
            }

            if (devicesList.Count <= 0)
            {
                Console.WriteLine("No devices found.");
                return;
            }

            _device = devicesList[0];

            if (Cl.GetDeviceInfo(_device, DeviceInfo.ImageSupport,
                      out error).CastTo<Bool>() == Bool.False)
            {
                Console.WriteLine("No image support.");
                return;
            }
            _context = Cl.CreateContext(null, 1, new[] { _device }, ContextNotify, IntPtr.Zero, out error);    //Second parameter is amount of devices
        }

        public static void Initialize()
        {
            //Initialize OpenCL
            Setup();

            _queue = Cl.CreateCommandQueue(_context, _device, CommandQueueProperties.None, out ErrorCode err);
        }

        public static void SubmitMemoryWrite(MemoryBuffer buf, int offset, int len, object data)
        {
            Cl.EnqueueWriteBuffer(_queue, buf.mem, Bool.False, (IntPtr)offset, (IntPtr)len, data, 0, null, out var ign);
        }

        public static void SubmitImage2DWrite(ComputeImage2D img, object data)
        {
            Cl.EnqueueWriteImage(_queue, img.img, Bool.False, new[] { IntPtr.Zero, IntPtr.Zero, IntPtr.Zero }, new[] { (IntPtr)img.Width, (IntPtr)img.Height, (IntPtr)1 }, IntPtr.Zero, IntPtr.Zero, data, 0, null, out var ign);
        }

        public static void SubmitMemoryRead(MemoryBuffer buf, int offset, int len, object data)
        {
            Cl.EnqueueReadBuffer(_queue, buf.mem, Bool.False, (IntPtr)offset, (IntPtr)len, data, 0, null, out var ign);
        }

        public static void SubmitImage2DRead(ComputeImage2D img, object dest)
        {
            Cl.EnqueueReadImage(_queue, img.img, Bool.False, new[] { IntPtr.Zero, IntPtr.Zero, IntPtr.Zero }, new[] { (IntPtr)img.Width, (IntPtr)img.Height, (IntPtr)1 }, IntPtr.Zero, IntPtr.Zero, dest, 0, null, out var ign);
        }

        public static void WaitFinish()
        {
            Cl.Finish(_queue);
        }

        public static void Barrier()
        {
            Cl.EnqueueBarrier(_queue);
        }

        public static void SubmitProgram(ComputeKernel prog, uint dim, IntPtr[] globalOffset, IntPtr[] globalSize, IntPtr[] localSize)
        {
            Cl.EnqueueNDRangeKernel(_queue, prog.kernel, dim, globalOffset, globalSize, localSize, 0, null, out var ign);
        }

        public static MemoryBuffer CreateMemory(int size, bool write, bool read)
        {
            var r = Cl.CreateBuffer(_context, (write && read ? MemFlags.ReadWrite : 0) | (write && !read ? MemFlags.WriteOnly : 0) | (!write && read ? MemFlags.ReadOnly : 0), size, out ErrorCode code);

            MemoryBuffer buf = new MemoryBuffer()
            {
                mem = r
            };

            return buf;
        }

        public static ComputeImage2D CreateImage2D(int w, int h, bool write, bool read)
        {
            ImageFormat fmt = new ImageFormat(ChannelOrder.RGB, ChannelType.Unsigned_Int8);
            var img = Cl.CreateImage2D(_context, (write && read ? MemFlags.ReadWrite : 0) | (write && !read ? MemFlags.WriteOnly : 0) | (!write && read ? MemFlags.ReadOnly : 0), fmt, (IntPtr)w, (IntPtr)h, IntPtr.Zero, null, out var err);

            ComputeImage2D cimg = new ComputeImage2D()
            {
                img = img,
                Width = w,
                Height = h
            };

            return cimg;
        }

        public static ComputeProgram CreateProgram(params string[] code)
        {
            var prog = Cl.CreateProgramWithSource(_context, 1, code, null, out var err);
            Cl.BuildProgram(prog, 1, new[] { _device }, "", null, IntPtr.Zero);
            if (Cl.GetProgramBuildInfo(prog, _device, ProgramBuildInfo.Status, out err).CastTo<BuildStatus>() != BuildStatus.Success)
            {
                Console.WriteLine("Program Build Failure!");
                throw new Exception();
            }
            
            ComputeProgram program = new ComputeProgram()
            {
                prog = prog
            };

            return program;
        }

        public static ComputeKernel CreateKernel(string name, ComputeProgram prog)
        {
            var kern = Cl.CreateKernel(prog.prog, name, out var err);

            ComputeKernel kernel = new ComputeKernel()
            {
                kernel = kern
            };

            return kernel;
        }
    }
}
