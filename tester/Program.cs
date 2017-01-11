using math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace tester
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            Matrix a = new Matrix(3, 1);
            a.FillRandom(-1.0, 1.0);
            Matrix b = new Matrix(3);
            b.FillRandom(1.0, 2.0);
            a.Print();
            b.Print();  
            Matrix p = a.Dot(b);
            p.Print();
            p = b.Dot(a);
            p.Print();

            Matrix c = new Matrix(2, 3);
            c.Map((x) => x + 1);
            c.Print();
            c.Map((x) => Math.Tanh(x));
            c.Print();
            Matrix.Transpose(c).Print();
            */
            Matrix x = new Matrix(3, 2);
            x.FillRandom(0.1, 1.1);
            x.Print("X=");
            //x.Row(1).Print("row(1)");
            //x.Column(1).Print("column(1)");

            Matrix y = new Matrix(3, 1);
            y.FillRandom(0.1, 1.0);
            y.Print("Y=");
            Matrix w0 = new Matrix(2, 2);
            w0.FillRandom(-0.5, 0.5);
            Matrix w1 = new Matrix(2, 1);
            w1.FillRandom(-0.5, 0.5);
            //
            Matrix e = null;
            Matrix a1 = null;
            for (int i = 0; i < x.Rows; i++)
            {
                w0.Print("w0=");
                Matrix a0 = x.Row(i).Dot(w0);
                a0.Map((v) => Math.Tanh(v));
                a0.Print("a0=");
                w1.Print("w1=");
                a1 = a0.Dot(w1);
                a1.Map((v) => Math.Tanh(v));
                a1.Print("a1=");
                e = y.Row(i) - a1;
                Console.WriteLine("Error = {0}", e.SquaredError());
            }
            // BP
            Matrix dO = a1;
            dO.Map((v) => e[0,0] * (1.0 - Math.Pow(Math.Tanh(v), 2)));
            dO.Print("dO=");
        }
    }
}
