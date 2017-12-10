using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utilities
{
    public class Logger
    {
        private TextWriter _f;
        private long _start;

        public delegate void PrintFn(TextWriter f, string msg);

        public Logger(string filename)
        {
            _f = File.CreateText(filename);
            _start = DateTime.Now.Ticks;
            Log("==== Start {0}", DateTime.UtcNow.ToString());
        }

        ~Logger()
        {
        }

        public void Log(string format, params object[] plist)
        {
            TimeSpan ts = new TimeSpan(DateTime.Now.Ticks - _start);
            _f.Write("{0,8:00000000} ", ts.TotalMilliseconds);
            _f.WriteLine(format, plist);
        }

        public void Log(PrintFn fn, string msg)
        {
            //TimeSpan ts = new TimeSpan(DateTime.Now.Ticks - _start);
            //_f.Write("{0,8:00000000} ", ts.TotalMilliseconds);
            //_f.WriteLine(msg);
            fn(_f, msg);
        }

        public void Flush()
        {
            _f.Flush();
        }
    }
}
