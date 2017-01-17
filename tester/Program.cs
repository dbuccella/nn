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
        const double Mu = 0.1;
        static double[,] X = new double[,] {
        {0.2, 1.020562258},
        {0.43, 1.218506908},
        {0.32, 0.184617471},
        {0.53, 1.534720884},
        {0.59, 0.822312634},
        {0.678, 0.419922912},
        {0.766, -0.216603169},
        {0.854, -0.583984585},
        {0.942, 1.698239806},
        {1.03, 0.953863041},
        {1.118, 1.875333287},
        {1.206, 1.132548225},
        {1.294, 2.058306355},
        {1.382, 0.293087256},
        {1.47, 0.444208613},
        {1.558, 1.892133761},
        {1.646, 0.99},
        {1.734, 0.93},
        {1.822, 0.988834473},
        {1.91, 0.948740604},
        {1.998, 0.908646735},
        {2.086, 0.868552867},
        {2.174, 0.828458998}};

        static double[,] Y = new double[,] {
        {0},
        {0},
        {1},
        {0},
        {0},
        {1},
        {1},
        {1},
        {0},
        {0},
        {0},
        {0},
        {0},
        {1},
        {1},
        {0},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1}};

        static double[,] VX = new double[,]{
        {0.473326958,0.032684313},
        {-0.257836836,0.170549968},
        {-0.851918897,-0.23161463},
        {0.989265156,0.744724906},
        {0.681344659,0.437958342},
        {0.710722887,0.636470473},
        {-0.562265989,-0.129908175},
        {-0.866112786,-0.231810514},
        {0.842265815,1.080677695},
        {0.03972403,0.614655844},
        {-0.939167934,-0.543522961},
        {-0.597955993,0.303809349},
        {-0.215915883,0.32955111},
        {-0.737554712,0.181623063},
        };

        static double[,] VY = new double[,] {
            {0},
            {1},
            {1},
            {1},
            {0},
            {0},
            {1},
            {1},
            {1},
            {1},
            {1},
            {0},
            {0},
            {1} };


        public static void Shuffle(int[] array)
        {
            Random gen = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < array.Length; i++)
            {
                int j = gen.Next(array.Length - i - 1);
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        static void Main3(string[] args)
        {
            Matrix x = new Matrix(X);
            Matrix y = new Matrix(Y);
            //y.Map((v) => (v == 0.0) ? -1.0 : 1.0);

            int Layers = 2;
            Matrix[] w = new Matrix[Layers];
            Matrix[] a = new Matrix[Layers];
            w[0] = new Matrix(2, 2);
            w[0].FillRandom(-1.0, 1.0);
            w[1] = new Matrix(2, 1);
            w[1].FillRandom(-1.0, 1.0);
            a[0] = null;
            a[1] = null;


            int[] idx = CreateIndex(x.Rows);
            int epoch = 0;
            double error = 1.0;

            while ((error > 0.1) && ((epoch < 10000)))
            {
                Matrix e = null;
                for (int i = 0; i < x.Rows; i++)
                {
                    // FF
                    a[0] = x.Row(idx[i]).Dot(w[0]).Map((v) => Math.Tanh(v));
                    a[1] = a[0].Dot(w[1]).Map((v) => Math.Tanh(v));
                    //
                    e = a[1].Transpose() - y.Row(idx[i]);
                    error = e.SquaredError();
                    //if ((epoch > 10000) || ((error < 0.1) && (epoch > 5)))
                    //    break;
                   
                    Matrix d2 = e * a[1].MapNew(Prime);
                    Matrix d1 = d2.Dot(w[1].Transpose()) * a[0].MapNew(Prime);
                    Matrix d0 = d1.Dot(w[0].Transpose()) * x.Row(idx[i]).MapNew(Prime);
                    //W[i] += s[i].T.dot(s_delta[i])
                    w[0] = w[0] + a[0].Transpose().Dot(d1).Multiply(Mu);
                    w[1].Map((v) => v + a[1][0, 0] * d2[0, 0] * Mu);
                    //w[1] = w[1] + a[1].Transpose().Dot(d2).Multiply(Mu);
                }
                Shuffle(idx);
                epoch++;
                if ((epoch % 50) == 0)
                    Console.WriteLine("=====> Error = {0}", error);
            }
            Console.WriteLine("=====> Last Error = {0:N8} epochs = {1}", error, epoch);
            w[0].Print("w0");
            w[1].Print("w1");
            // verify
            Matrix vx = new Matrix(VX);
            Matrix vy = new Matrix(VY);
            //vy.Map((v) => (v == 0.0) ? -1.0 : 1.0);
            for (int i = 0; i < vx.Rows; i++)
            {
                a[0] = x.Row(i).Dot(w[0]).Map((v) => Math.Tanh(v));
                a[1] = a[0].Dot(w[1]).Map((v) => Math.Tanh(v));
                Console.WriteLine("target = {0} | actual = {1}", a[1][0, 0], vy[i, 0]);
            }
        }

        static int [] CreateIndex(int sz)
        {
            int[] idx = new int[sz];
            for (int i = 0; i < sz; i++)
                idx[i] = i;
            return idx;
        }

        static void Main2(string[] args)
        {
            Matrix x = new Matrix(X);
            Matrix y = new Matrix(Y);

            Matrix w0 = new Matrix(2, 2);
            w0.FillRandom(-0.5, 0.5);
            Matrix w1 = new Matrix(2, 1);
            w1.FillRandom(-0.5, 0.5);
            //
            int[] idx = new int[x.Rows];
            for (int i = 0; i < x.Rows; i++)
                idx[i] = i;
            Matrix a1 = null;
            double error = 1.0;
            int iter = 0;
            while ((iter < 10000) && (error > 0.00001))
            {
                for (int i = 0; i < x.Rows; i++)
                {
                    Matrix a0 = w0.Dot(x.Row(idx[i]).Transpose());
                    a0.Map((v) => Math.Tanh(v));
                    a1 = w1.Dot(a0.Transpose());
                    a1.Map((v) => Math.Tanh(v));
                    Matrix e = y.Row(idx[i]) - a1.Transpose();
                    error = e.SquaredError();
                    // BP
                    Matrix d2 = e * a1.MapNew(Prime);
                    Matrix d1 = d2.Transpose().Dot(w1) * a0.MapNew(Prime);
                    w0 = w0 + d1.Transpose().Dot(a0).Map((v) => v * Mu);
                    w1 = w1 + d2.Transpose().Dot(a1).Map((v) => v * Mu);
                    Console.WriteLine("=====> Error = {0}", error);
                    /*


                    //w0.Print("w0=");
                    Matrix a0 = x.Row(idx[i]).Dot(w0);
                    a0.Map((v) => Math.Tanh(v));
                    //a0.Print("a0=");
                    //w1.Print("w1=");
                    a1 = a0.Dot(w1);
                    a1.Map((v) => Math.Tanh(v));
                    //a1.Print("a1=");
                    Matrix e = y.Row(idx[i]) - a1;
                    //Console.WriteLine();
                    error = e.SquaredError();
                    Console.WriteLine("=====> Error = {0}", error);
                    //
                    // BP
                    Matrix d2 = e * a1.MapNew(Prime);
                    //d2.Print("d2=");
                    Matrix d1 = d2.Dot(w1.Transpose()) * a0.MapNew(Prime);
                    //d1.Print("d1=");
                    Matrix d0 = d1.Dot(w0.Transpose()) * x.Row(idx[i]).MapNew(Prime);
                    //d0.Print("d0=");
                    // update weights
                    w0 = w0 + d0.Transpose().Dot(a0).Map((v) => v * Mu);
                    //w0.Print("w0=");
                    w1 = w1 + d1.Transpose().Dot(a1);
                    //w1.Print("w1=");
                    */
                }
                Shuffle(idx);
            }
        }

        public static double Prime(double x)
        {
            return (1.0 - Math.Pow(Math.Tanh(x), 2.0));
        }

        static void Main(string[] args)
        {
            mlp net = new mlp();
            Matrix x = new Matrix(X);
            Matrix y = new Matrix(Y);
            net.Train(x, y);
            Matrix vx = new Matrix(VX);
            Matrix vy = new Matrix(VY);
            net.Verify(vx, vy);
        }


    }
}
